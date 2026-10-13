"""Qwen3-VL-4B detection on the tiny GT crop with a green background.

Run in the qwen3-vl environment:
    python infer.py
    python infer.py --dry-run  # inspect inputs without loading the model

Input is fixed to edited_images/1dc20e8b1/1dc20e8b1_crop_tiny_greenbackground.png.
Detection visualizations are saved as <image_stem>_vis.png under
edited_images/1dc20e8b1/vis_tiny_nowaves/exp3.
Complete generated responses, including special tokens, are saved as UTF-8
text files under responses/ before detection parsing, even if parsing fails.
Dry-run previews are saved separately in that output directory's dry_run folder.

Model bbox_2d coordinates are xyxy normalized to 0..1000.
Saved bbox_xyxy coordinates refer to the original image in pixels.
Metrics use class-aware greedy IoU matching, not COCO mAP.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time

from PIL import Image, ImageDraw, ImageFont
#LZ

DEFAULT_MODEL = Path("/data/lz/models/Qwen3-VL-4B-Instruct")
DEFAULT_IMAGES = Path("/data/lz/pythonProjects/qwen3-vl/Qwen3-VL/edited_images/1dc20e8b1")
DEFAULT_IMAGE = DEFAULT_IMAGES / "1dc20e8b1_crop_tiny_greenbackground.png"
DEFAULT_OUTPUT = DEFAULT_IMAGES / "vis_tiny_nowaves" / "exp3"
CATEGORIES = (
    "airplane", "bridge", "storage-tank", "ship",
    "swimming-pool", "vehicle", "person", "wind-mill",
)
COLORS = ("#0000ff",)


def canonical_label(label):
    return re.sub(r"[\s_]+", "-", str(label).strip().lower())


def build_prompt(categories):
    return (
        "This is a satellite or aerial remote-sensing image. Locate every "
        "individual instance belonging to these categories: "
        + ", ".join(categories)
        + ". Pay attention to tiny objects. "
        "Draw a tight box around each individual object; do not combine nearby "
        "objects into one box. Use only the exact category names above. "
        'Return only a JSON array of {"bbox_2d": [x1, y1, x2, y2], '
        '"label": "category"}. Coordinates must be normalized to 0-1000 relative '
        "to this image, in top-left to bottom-right order. "
    )


def select_images():
    if not DEFAULT_IMAGE.is_file():
        raise ValueError(f"裁剪图像不存在: {DEFAULT_IMAGE}")
    return [DEFAULT_IMAGE.resolve()]


def load_ground_truth(annotation_path, paths, categories):
    """COCO xywh boxes are converted to original-image xyxy coordinates."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    names = {c["id"]: canonical_label(c["name"]) for c in data["categories"]}
    selected = {p.name for p in paths}
    images = {im["id"]: im for im in data["images"] if im["file_name"] in selected}
    result = {im["file_name"]: {"image_id": im["id"], "boxes": []}
              for im in images.values()}
    for annotation in data["annotations"]:
        image = images.get(annotation["image_id"])
        label = names[annotation["category_id"]]
        if image is None or label not in categories:
            continue
        if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
            raise ValueError("所选标注含 crowd/ignore，需要用 COCO evaluator 评测")
        x, y, w, h = annotation["bbox"]
        result[image["file_name"]]["boxes"].append(
            {"label": label, "bbox_xyxy": [x, y, x + w, y + h]}
        )
    return result


def parse_detections(text, width, height, categories=CATEGORIES):
    """Accept JSON fences/preamble; malformed output must not become valid []."""
    fence = chr(96) * 3
    match = re.search(fence + r"(?:json)?\s*([\s\S]*?)" + fence, text, re.IGNORECASE)
    payload = match.group(1).strip() if match else text.strip()
    start = re.search(r"[\[{]", payload)
    if start is None:
        raise ValueError("模型输出中没有 JSON 数组或对象")
    try:
        data, _ = json.JSONDecoder().raw_decode(payload[start.start():])
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出不是完整 JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("detections", [data])
    if not isinstance(data, list):
        raise ValueError("模型输出必须是检测结果数组")
    detections, warnings = [], []
    for i, item in enumerate(data):
        try:
            if not isinstance(item, dict):
                raise ValueError("检测项不是对象")
            label = canonical_label(item.get("label", ""))
            if label not in categories:
                raise ValueError(f"未知类别 {label!r}")
            box = item.get("bbox_2d")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("bbox_2d 必须包含 4 个坐标")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
                raise ValueError("坐标必须是数字")
            if any(not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
                raise ValueError("坐标必须在 0-1000 范围内")
            x1, x2 = sorted((float(box[0]), float(box[2])))
            y1, y2 = sorted((float(box[1]), float(box[3])))
            if x1 == x2 or y1 == y2:
                raise ValueError("边界框面积为零")
            detections.append({
                "label": label, "bbox_2d": [x1, y1, x2, y2],
                "bbox_xyxy": [x1 * width / 1000, y1 * height / 1000,
                              x2 * width / 1000, y2 * height / 1000],
            })
        except ValueError as exc:
            warnings.append(f"第 {i + 1} 项已跳过: {exc}")
    if data and not detections:
        raise ValueError("全部检测项无效: " + "; ".join(warnings))
    return detections, warnings


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
    return overlap / union if union > 0 else 0.0


def detection_counts(predictions, truth, categories, threshold):
    """Match each prediction/GT at most once, in descending IoU order."""
    result = {}
    for label in categories:
        pred = [p["bbox_xyxy"] for p in predictions if p["label"] == label]
        gt = [g["bbox_xyxy"] for g in truth if g["label"] == label]
        candidates = []
        for i, p in enumerate(pred):
            for j, g in enumerate(gt):
                iou = box_iou(p, g)
                if iou >= threshold:
                    candidates.append((iou, i, j))
        used_pred, used_gt = set(), set()
        for _, i, j in sorted(candidates, reverse=True):
            if i not in used_pred and j not in used_gt:
                used_pred.add(i)
                used_gt.add(j)
        tp = len(used_pred)
        result[label] = {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}
    return result


def add_rates(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {**counts, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def draw_boxes(image, boxes, categories):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(categories)}
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        color = colors[box["label"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), box["label"], fill=color, font=font)
    return image


def save_visualization(image, predictions, categories, path):
    draw_boxes(image, predictions, categories).save(path)


def validate_model_files(model_path):
    if not model_path.is_dir():
        raise ValueError(f"本地模型目录不存在: {model_path}")
    required = ["config.json", "tokenizer_config.json", "tokenizer.json",
                "preprocessor_config.json"]
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        required += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        required.append("model.safetensors")
    missing, pointers = [], []
    for name in required:
        path = model_path / name
        if not path.is_file():
            missing.append(name)
        else:
            with path.open("rb") as handle:
                if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    pointers.append(name)
    if missing or pointers:
        raise ValueError(
            "模型文件未下载完整。缺失: " + (", ".join(missing) or "无")
            + "；Git LFS 占位文件: " + (", ".join(pointers) or "无")
            + "。请补齐模型文件后再推理；--dry-run 可先检查数据。"
        )


def load_model(args):
    validate_model_files(args.model_path)
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU/驱动；CPU 推理需显式指定 --device cpu")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), local_files_only=True,
        dtype=torch.float32 if args.device == "cpu" else "auto",
        device_map=args.device, attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    return model, processor


def infer_image(model, processor, image, prompt, max_new_tokens):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_ids = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    full_text = processor.batch_decode(
        output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )[0]
    return text, full_text, output_ids.shape[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"只检测 {DEFAULT_IMAGE}，并将可视化图保存到 {DEFAULT_OUTPUT}"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--annotations", type=Path, help="可选 COCO 标注，仅用于指标统计")
    parser.add_argument("--no-ground-truth", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="只验证数据/标注，不加载模型")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens 必须为正")
    if not 0 < args.min_pixels <= args.max_pixels or not 0 < args.iou_threshold <= 1:
        parser.error("需满足 0 < min-pixels <= max-pixels，且 0 < iou-threshold <= 1")
    args.categories = list(dict.fromkeys(canonical_label(c) for c in args.categories))
    if not all(args.categories):
        parser.error("类别名称不能为空")
    return args


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    paths = select_images()
    annotations = args.annotations
    ground_truth = {}
    if annotations is not None and not args.no_ground_truth:
        ground_truth = load_ground_truth(annotations, paths, args.categories)
        missing = [p.name for p in paths if p.name not in ground_truth]
        if missing:
            print(f"提示：{len(missing)} 张图像没有匹配标注，不参与指标统计", flush=True)
    prompt = build_prompt(args.categories)
    mode = "数据检查（无模型推理）" if args.dry_run else "模型推理"
    print(f"选中 {len(paths)} 张图像；模式: {mode}", flush=True)
    model, processor = (None, None) if args.dry_run else load_model(args)
    output_dir = DEFAULT_OUTPUT / "dry_run" if args.dry_run else DEFAULT_OUTPUT
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "image_dir": str(DEFAULT_IMAGES), "image": str(DEFAULT_IMAGE),
        "output_dir": str(output_dir),
        "resolved_annotations": str(annotations) if annotations else None,
        "prompt": prompt, "selected_images": [str(p) for p in paths],
    })
    records, totals = [], defaultdict(Counter)
    evaluated = 0
    for index, path in enumerate(paths, 1):
        record = {"image": str(path), "status": "dry_run" if args.dry_run else "ok"}
        started = time.perf_counter()
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            record["width"], record["height"] = image.size
            gt = ground_truth.get(path.name)
            if gt is not None:
                record.update(image_id=gt["image_id"], ground_truth=gt["boxes"])
            predictions = []
            if not args.dry_run:
                raw_text, full_text, tokens = infer_image(
                    model, processor, image, prompt, args.max_new_tokens
                )
                response_path = output_dir / "responses" / f"{path.stem}_response.txt"
                record.update(raw_output=raw_text, full_output=full_text,
                              response_file=str(response_path), generated_tokens=tokens,
                              generation_limit_reached=tokens >= args.max_new_tokens)
                response_path.write_text(full_text, encoding="utf-8")
                predictions, warnings = parse_detections(raw_text, *image.size, args.categories)
                record.update(detections=predictions, warnings=warnings)
                if record["generation_limit_reached"]:
                    record["warnings"].append("达到生成长度上限，结果可能不完整")
            visual_path = output_dir / f"{path.stem}_vis.png"
            save_visualization(image, predictions, args.categories, visual_path)
            record["visualization"] = str(visual_path)
            if not args.dry_run and gt is not None:
                counts = detection_counts(predictions, gt["boxes"], args.categories, args.iou_threshold)
                record["metrics"] = {label: add_rates(c) for label, c in counts.items()}
                for label, counts_for_label in counts.items():
                    totals[label].update(counts_for_label)
                evaluated += 1
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(output_dir / "results" / f"{path.name}.json", record)
        records.append(record)
        detail = record.get("error", f"GT={len(record.get('ground_truth', []))}, "
                            f"pred={len(record.get('detections', []))}")
        print(f"[{index}/{len(paths)}] {path.name}: {record['status']}, {detail}", flush=True)
    summary = {
        "mode": "dry_run" if args.dry_run else "inference", "selected_images": len(paths),
        "successful_images": sum(r["status"] != "error" for r in records),
        "failed_images": sum(r["status"] == "error" for r in records),
        "evaluated_images": evaluated, "iou_threshold": args.iou_threshold,
        "metric_note": "Class-aware greedy IoU matching; NOT COCO mAP. "
                       "Only successful annotated images are evaluated; inspect failures/warnings.",
        "per_category": {label: add_rates(dict(c)) for label, c in totals.items()},
        "images": [{k: r[k] for k in ("image", "status", "elapsed_seconds", "error") if k in r}
                   for r in records],
    }
    if evaluated:
        summary["overall"] = add_rates({k: sum(c[k] for c in totals.values())
                                       for k in ("tp", "fp", "fn")})
        print("总体指标:", json.dumps(summary["overall"], ensure_ascii=False))
    write_json(output_dir / "summary.json", summary)
    print(f"结果目录: {output_dir.resolve()}")
    return 1 if summary["failed_images"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)


