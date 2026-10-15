"""COCO-style bbox metrics in ORIGINAL image coordinates (not official AI-TOD bins)."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .criterion import cxcywh_to_xyxy
from .data import CocoDetectionData, ImageCollator, move_batch


@torch.no_grad()
def to_coco_predictions(output, targets, categories):
    probabilities = output["pred_logits"].float().softmax(-1)[..., :-1]
    scores, labels = probabilities.max(-1)
    boxes = cxcywh_to_xyxy(output["pred_boxes"].float()).clamp(0, 1)
    results = []
    for batch_index, target in enumerate(targets):
        w, h = target["original_wh"]
        scaled = boxes[batch_index] * boxes.new_tensor([w, h, w, h])
        scaled[:, 2:] -= scaled[:, :2]
        for score, label, box in zip(scores[batch_index].cpu().tolist(),
                                     labels[batch_index].cpu().tolist(), scaled.cpu().tolist()):
            # Retain low-confidence predictions for AP; thresholds are ONLY for viewing.
            results.append({"image_id": target["image_id"], "category_id": categories[label]["id"],
                            "bbox": box, "score": score})
    return results


def coco_metrics(dataset, predictions, max_detections):
    ground_truth = COCO(str(dataset.annotation_path))
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {"images": deepcopy(ground_truth.dataset["images"]),
                              "categories": deepcopy(ground_truth.dataset["categories"]),
                              "annotations": []}
        detections.createIndex()
    result = {}
    for maximum in sorted({100, max(100, max_detections)}):
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = [image["id"] for image in dataset.images]
        evaluator.params.catIds = [category["id"] for category in dataset.categories]
        evaluator.params.maxDets = [1, 10, maximum]
        evaluator.evaluate()
        evaluator.accumulate()
        precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]

        def mean_valid(values):
            values = values[values > -1]
            return float(values.mean()) if values.size else None

        def ap(area=0, iou=None):
            values = precision[:, :, :, area, -1]
            if iou is not None:
                values = values[np.isclose(evaluator.params.iouThrs, iou)]
            return mean_valid(values)

        result[f"coco_maxDets_{maximum}"] = {
            "AP": ap(), "AP50": ap(iou=.5), "AP75": ap(iou=.75),
            "AP_small": ap(area=1), "AP_medium": ap(area=2), "AP_large": ap(area=3),
            "AR": mean_valid(recall[:, :, 0, -1]),
            "per_class_AP": {category["name"]: mean_valid(precision[:, :, i, 0, -1])
                             for i, category in enumerate(dataset.categories)},
        }
    return {"protocol": "COCO bbox IoU .50:.05:.95; COCO area ranges; one class per query; no NMS",
            "scale": "AP is 0..1; null means no eligible ground truth", **result}


def save_views(dataset, predictions, directory, threshold=.2, count=4):
    """Small visual check only; threshold NEVER affects the AP predictions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selected = {im["id"] for im in dataset.images[:count]}
    by_image = {image_id: [] for image_id in selected}
    for prediction in predictions:
        if prediction["image_id"] in selected and prediction["score"] >= threshold:
            by_image[prediction["image_id"]].append(prediction)
    names = {c["id"]: c["name"] for c in dataset.categories}
    for index, info in enumerate(dataset.images[:count]):
        image, target = dataset[index]
        w, h = image.size
        canvas = Image.new("RGB", (w * 2, h + 30), "white")
        canvas.paste(image, (0, 30))
        canvas.paste(image, (w, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), "Ground truth", fill="black")
        draw.text((w + 5, 5), f"Predictions score >= {threshold} (top 100 for display)", fill="black")
        boxes = cxcywh_to_xyxy(target["boxes"]) * torch.tensor([w, h, w, h])
        for box, label in zip(boxes.tolist(), target["labels"].tolist()):
            x0, y0, x1, y1 = box
            draw.rectangle((x0, y0 + 30, x1, y1 + 30), outline="lime", width=1)
            draw.text((x0, max(30, y0 + 18)), dataset.categories[label]["name"], fill="lime")
        for prediction in sorted(by_image[info["id"]], key=lambda p: p["score"], reverse=True)[:100]:
            x, y, bw, bh = prediction["bbox"]
            draw.rectangle((x + w, y + 30, x + w + bw, y + 30 + bh), outline="red", width=1)
            draw.text((x + w, max(30, y + 18)),
                      f"{names[prediction['category_id']]} {prediction['score']:.2f}", fill="red")
        canvas.save(directory / f"image_{info['id']}.png")


@torch.no_grad()
def evaluate(model, loader, dataset, device, precision, output_path):
    model.eval()
    predictions = []
    for inputs, targets in loader:
        inputs, targets = move_batch(inputs, targets, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            outputs = model(**inputs)
        if not torch.isfinite(outputs["pred_logits"]).all() or not torch.isfinite(outputs["pred_boxes"]).all():
            raise FloatingPointError("Non-finite detection output")
        predictions.extend(to_coco_predictions(outputs, targets, dataset.categories))
    Path(output_path).write_text(json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")
    save_views(dataset, predictions, Path(output_path).parent / "visualizations_last")
    return coco_metrics(dataset, predictions, model.head.num_queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor
    from .detection import build_detector

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.precision == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("Use --precision float32 on CPU or unsupported GPUs")
    data = CocoDetectionData(args.images, args.annotations, categories=checkpoint["categories"])
    processor = AutoImageProcessor.from_pretrained(config["vit_path"], local_files_only=True,
                                                   **config["preprocessing"])
    loader = DataLoader(data, batch_size=config["training"]["batch_size"], collate_fn=ImageCollator(processor))
    model = build_detector(config["vit_path"], config["head"], frozen=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device)
    metrics = evaluate(model, loader, data, device, args.precision, output_dir / "predictions.json")
    report = {"checkpoint": args.checkpoint, "dataset": data.summary(), "metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
