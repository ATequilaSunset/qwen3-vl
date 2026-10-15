"""Convert VisDrone MOT GT to per-frame detection diagnostics, never tracking scores.

Input: sequences/<sequence>/<frame>.jpg and annotations/<sequence>.txt.
Rows: frame,id,left,top,width,height,score,category,truncation,occlusion.
Ignore regions are retained for training and represented by per-class COCO
crowd boxes for evaluation. This is a custom diagnostic, not official VisDrone.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import re

from PIL import Image, ImageDraw

from .prepare_coco import sha256, write_json


CATEGORIES = [{"id": i, "name": name} for i, name in enumerate(
    ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"], 1)]


def video_group(sequence):
    # Conservatively group multiple clips named uavNNNNNNN_* together.
    match = re.match(r"^(uav\d+)_", sequence, flags=re.IGNORECASE)
    return match.group(1).lower() if match else sequence


def read_mot(path):
    frames = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            while row and not row[-1].strip():
                row.pop()
            if not row:
                continue
            if len(row) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10-column MOT ground truth, got {len(row)}")
            values = [float(value.strip()) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite annotation at {path}:{line_number}")
            frame, track, x, y, w, h, score, category, truncation, occlusion = values
            if any(values[i] != int(values[i]) for i in (0, 1, 6, 7, 8, 9)):
                raise ValueError(f"Noninteger MOT index/category/flag at {path}:{line_number}")
            if frame < 1 or score not in (0, 1) or not 0 <= category <= 11:
                raise ValueError(f"Invalid ground-truth frame/score/category at {path}:{line_number}")
            frames[int(frame)].append({"bbox": [x, y, w, h], "track_id": int(track),
                                       "source_score": int(score), "source_category": int(category),
                                       "truncation": int(truncation), "occlusion": int(occlusion)})
    return frames


def discover(root):
    root = Path(root).expanduser().resolve()
    sequence_root, annotation_root = root / "sequences", root / "annotations"
    if not sequence_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(f"Expected {sequence_root}/<sequence>/*.jpg and {annotation_root}/<sequence>.txt")
    entries = []
    for folder in sorted(p for p in sequence_root.iterdir() if p.is_dir()):
        annotation = annotation_root / f"{folder.name}.txt"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing MOT annotations: {annotation}")
        frames = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if not path.stem.isdecimal() or int(path.stem) < 1:
                raise ValueError(f"Expected positive numeric frame filename: {path}")
            frame = int(path.stem)
            if frame in frames:
                raise ValueError(f"Duplicate numeric frame index in {folder}")
            frames[frame] = path
        if not frames:
            raise ValueError(f"No image frames in {folder}")
        entries.append({"name": folder.name, "group": video_group(folder.name), "frames": frames,
                        "annotations": annotation, "sequence_root": sequence_root})
    if not entries:
        raise ValueError(f"No sequences in {sequence_root}")
    return entries


def split_groups(entries, fraction, seed):
    groups = sorted({entry["group"] for entry in entries})
    if len(groups) < 2:
        return entries, []
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    held_out = set(groups[:count])
    return ([entry for entry in entries if entry["group"] not in held_out],
            [entry for entry in entries if entry["group"] in held_out])


def clip_bbox(bbox, width, height):
    x, y, w, h = bbox
    if min(w, h) <= 0:
        return None
    x0, y0, x1, y1 = max(0., x), max(0., y), min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def convert(entries, stride):
    document = {"info": {"description": "VisDrone MOT -> per-frame COCO diagnostic; not official MOT/DET metrics",
                          "ignore_protocol": "Class-agnostic ignore regions; copied as COCO crowds for all categories"},
                "images": [], "annotations": [], "categories": deepcopy(CATEGORIES)}
    audit = {"sequences": [], "counts": Counter(), "positive_category_counts": Counter()}
    for entry in entries:
        annotations = read_mot(entry["annotations"])
        unknown_frames = set(annotations) - set(entry["frames"])
        if unknown_frames:
            raise ValueError(f"{entry['annotations']} references missing image frames: {sorted(unknown_frames)[:5]}")
        audit["sequences"].append({"name": entry["name"], "video_group": entry["group"],
                                    "annotation_sha256": sha256(entry["annotations"]),
                                    "source_frames": len(entry["frames"])})
        for frame, path in sorted(entry["frames"].items()):
            if (frame - 1) % stride:
                continue
            with Image.open(path) as image:
                width, height = image.size
            image_id = len(document["images"]) + 1
            info = {"id": image_id, "file_name": path.relative_to(entry["sequence_root"]).as_posix(),
                    "width": width, "height": height, "sequence_id": entry["name"],
                    "video_group": entry["group"], "frame_index": frame, "ignore_regions_xywh": [],
                    "ignored_source_objects": []}
            positives = []
            for obj in annotations.get(frame, []):
                ignored = obj["source_score"] == 0 or obj["source_category"] in (0, 11)
                bbox = clip_bbox(obj["bbox"], width, height)
                if bbox is None:
                    if ignored:
                        audit["counts"]["empty_ignored_boxes_skipped"] += 1
                        continue
                    raise ValueError(f"Invalid positive bbox: {entry['name']}, frame {frame}, {obj}")
                if bbox != obj["bbox"]:
                    audit["counts"]["clipped_source_boxes"] += 1
                if ignored:
                    info["ignore_regions_xywh"].append(bbox)
                    info["ignored_source_objects"].append(deepcopy(obj))
                    audit["counts"]["ignore_regions"] += 1
                    continue
                positives.append({"image_id": image_id, "category_id": obj["source_category"],
                                  "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                                  "track_id": obj["track_id"], "truncation": obj["truncation"],
                                  "occlusion": obj["occlusion"], "source_bbox": obj["bbox"]})
                audit["positive_category_counts"][obj["source_category"]] += 1
            for ann in positives:
                document["annotations"].append({"id": len(document["annotations"]) + 1, **ann})
            # COCO crowd matching uses intersection over detection area, handles
            # multiple predictions, and prioritizes ordinary valid GT matches.
            for bbox in info["ignore_regions_xywh"]:
                for category in CATEGORIES:
                    document["annotations"].append({"id": len(document["annotations"]) + 1,
                                                    "image_id": image_id, "category_id": category["id"],
                                                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 1,
                                                    "visdrone_ignore_region": True})
            document["images"].append(info)
            audit["counts"]["sampled_frames"] += 1
            audit["counts"]["positive_objects"] += len(positives)
            audit["counts"]["frames_without_positive_objects"] += not positives
    if not document["images"]:
        raise ValueError("No frames after conversion/stride selection")
    return document, audit


def select_fitting(document, count, max_objects, seed):
    counts = Counter(a["image_id"] for a in document["annotations"] if not a["iscrowd"])
    grouped = defaultdict(list)
    for info in document["images"]:
        if 1 <= counts[info["id"]] <= max_objects:
            grouped[info["video_group"]].append(info)
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(grouped[key])
    picked = []
    while len(picked) < count:
        changed = False
        for key in keys:
            if grouped[key] and len(picked) < count:
                picked.append(grouped[key].pop())
                changed = True
        if not changed:
            raise ValueError(f"Only {len(picked)} eligible frames; need {count}. "
                             "Increase --max-objects or decrease --frame-stride. No GT was dropped.")
    ids = {info["id"] for info in picked}
    return {**deepcopy(document), "images": sorted(picked, key=lambda im: im["id"]),
            "annotations": [a for a in document["annotations"] if a["image_id"] in ids]}


def save_views(document, root, directory):
    directory.mkdir()
    objects = defaultdict(list)
    for ann in document["annotations"]:
        if not ann["iscrowd"]:
            objects[ann["image_id"]].append(ann)
    names = {category["id"]: category["name"] for category in CATEGORIES}
    for info in document["images"]:
        with Image.open(root / info["file_name"]) as original:
            image = original.convert("RGB")
        draw = ImageDraw.Draw(image)
        for bbox in info["ignore_regions_xywh"]:
            x, y, w, h = bbox
            draw.rectangle((x, y, x + w, y + h), outline="orange", width=2)
            draw.text((x, max(0, y - 12)), "IGNORE", fill="orange")
        for ann in objects[info["id"]]:
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="lime", width=2)
            draw.text((x, max(0, y - 12)), names[ann["category_id"]], fill="lime")
        image.save(directory / f"{info['sequence_id']}_{info['frame_index']:07d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/lz/VisDrone2019-MOT-train")
    parser.add_argument("--val-root", help="Optional official MOT validation root; otherwise hold out video groups")
    parser.add_argument("--output-dir", default="vit_probe/artifacts/visdrone_sanity")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/coco_detection.json"))
    parser.add_argument("--vit-path")
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=100, help="Exclude entire dense frames only for fitting selection")
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.images <= 32 or min(args.frame_stride, args.max_objects) < 1 or not 0 < args.val_fraction < 1:
        parser.error("Use 1..32 fitting images, positive stride/capacity and val-fraction in (0,1)")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    entries = discover(args.root)
    if args.val_root:
        train_entries, val_entries = entries, discover(args.val_root)
        if {e["group"] for e in train_entries} & {e["group"] for e in val_entries}:
            raise ValueError("Train/val share video groups; inspect dataset provenance")
        split_mode = "provided_validation_root"
    else:
        train_entries, val_entries = split_groups(entries, args.val_fraction, args.seed)
        split_mode = "held_out_video_groups_within_training_release"
    train, train_audit = convert(train_entries, args.frame_stride)
    val, val_audit = convert(val_entries, args.frame_stride) if val_entries else (None, None)
    fitting = select_fitting(train, args.images, args.max_objects, args.seed)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("Expected schema=1 detection config")
    config["training"]["seed"] = args.seed
    if args.vit_path:
        config["vit_path"] = args.vit_path
    config["vit_path"] = str(Path(config["vit_path"]).expanduser().resolve())
    train_root = train_entries[0]["sequence_root"]
    config["data"] = {"train_images": str(train_root), "train_annotations": str(output / "train_annotations.json"),
                      "val_images": str(val_entries[0]["sequence_root"]) if val_entries else None,
                      "val_annotations": str(output / "val_annotations.json") if val is not None else None}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train_annotations.json", train)
    write_json(output / "overfit_annotations.json", fitting)
    if val is not None:
        write_json(output / "val_annotations.json", val)
        write_json(output / "visdrone_pilot.json", config)
    fitting_config = deepcopy(config)
    fitting_config["data"]["train_annotations"] = str(output / "overfit_annotations.json")
    write_json(output / "visdrone_overfit.json", fitting_config)
    save_views(fitting, train_root, output / "ground_truth")
    counts = Counter(a["image_id"] for a in fitting["annotations"] if not a["iscrowd"])
    report = {"task": "per-frame detection; tracking identities not used by model", "split_mode": split_mode,
              "root": str(Path(args.root).resolve()), "stride": args.frame_stride, "seed": args.seed,
              "categories": CATEGORIES, "train": train_audit, "validation": val_audit,
              "fitting_frames": [{"image_id": im["id"], "sequence": im["sequence_id"], "frame": im["frame_index"],
                                  "positive_objects": counts[im["id"]], "ignore_regions": len(im["ignore_regions_xywh"])}
                                 for im in fitting["images"]],
              "max_objects_filter_applies_only_to_fitting": args.max_objects,
              "all_positive_annotations_kept": True, "ignore_policy": "score=0/category=0,11 -> class-agnostic ignore; "
              "unmatched queries with intersection/prediction area >=0.5 have no background loss; "
              "evaluation uses per-category COCO crowd regions (custom protocol)",
              "vit_path": config["vit_path"], "vit_weights_present": (Path(config["vit_path"]) / "vit.safetensors").is_file(),
              "overfit_config": str(output / "visdrone_overfit.json"),
              "pilot_config": str(output / "visdrone_pilot.json") if val is not None else None,
              "note": "10-class vocabulary retains score=1 annotations among IDs 1..10; classes absent in this MOT release "
                      "have no GT. This is not the official five-class MOT benchmark. No images are copied or resized here."}
    write_json(output / "prepare_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
