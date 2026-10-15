"""Prepare reproducible COCO easy-image fitting and optional independent validation.

Select images, never individual annotations: all GT for selected images is kept.
The easy subset is only a training-pipeline diagnostic, not a COCO benchmark.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Detection requires instances JSON, not images alone. "
                                "Use --train-annotations / --val-annotations for nonstandard locations.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_document(document):
    categories = sorted(document["categories"], key=lambda item: item["id"])
    if len(categories) != 80 or len({c["id"] for c in categories}) != 80:
        raise ValueError("Expected the 80-category COCO instances annotations; check the JSON file")
    images = {im["id"]: im for im in document["images"]}
    if not images or len(images) != len(document["images"]):
        raise ValueError("Image IDs must be nonempty and unique")
    category_ids = {c["id"] for c in categories}
    annotations = defaultdict(list)
    seen = set()
    for ann in document["annotations"]:
        if ann["id"] in seen or ann["image_id"] not in images or ann["category_id"] not in category_ids:
            raise ValueError("Duplicate annotation ID or unknown image/category ID")
        seen.add(ann["id"])
        annotations[ann["image_id"]].append(ann)
    return images, annotations, categories


def select_easy(document, count, min_box_side, max_objects, seed):
    images, annotations, _ = check_document(document)
    eligible = []
    for image_id, info in sorted(images.items()):
        objects = annotations[image_id]
        if not 1 <= len(objects) <= max_objects:
            continue
        valid = True
        for ann in objects:
            if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                valid = False
                break
            x, y, w, h = ann["bbox"]
            # Exclude the whole image if any GT is small/invalid/cropped out.
            # Keep every annotation unchanged on images that survive.
            if (not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0
                    or min(info["width"], info["height"]) <= 0):
                valid = False
                break
            visible_w = min(info["width"], x + w) - max(0, x)
            visible_h = min(info["height"], y + h) - max(0, y)
            if min(visible_w, visible_h) < min_box_side:
                valid = False
                break
        if valid:
            eligible.append(image_id)
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible images for requested {count}; "
                         "inspect data or explicitly relax --min-box-side / --max-objects")
    selected = set(random.Random(seed).sample(eligible, count))
    subset = {key: deepcopy(value) for key, value in document.items()
              if key not in ("images", "annotations")}
    subset["images"] = [images[i] for i in sorted(selected)]
    subset["annotations"] = [ann for ann in document["annotations"] if ann["image_id"] in selected]
    return subset, len(eligible)


def image_path(root, info):
    root = Path(root).resolve()
    path = (root / info["file_name"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Image missing or outside image root: {path}")
    return path


def save_gt(subset, root, directory):
    names = {c["id"]: c["name"] for c in subset["categories"]}
    by_image = defaultdict(list)
    for ann in subset["annotations"]:
        by_image[ann["image_id"]].append(ann)
    directory.mkdir()
    for info in subset["images"]:
        with Image.open(image_path(root, info)) as source:
            if source.size != (info["width"], info["height"]):
                raise ValueError(f"Image dimensions disagree with annotations: {info['file_name']}")
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for ann in by_image[info["id"]]:
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="lime", width=2)
            draw.text((max(0, x), max(0, y - 12)), names[ann["category_id"]], fill="lime")
        image.save(directory / f"image_{info['id']}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/coco_detection.json"))
    parser.add_argument("--output-dir", default="vit_probe/artifacts/coco_sanity")
    parser.add_argument("--vit-path")
    parser.add_argument("--train-images")
    parser.add_argument("--train-annotations")
    parser.add_argument("--val-images")
    parser.add_argument("--val-annotations")
    parser.add_argument("--images", type=int, default=4, help="Number of easy fitting images (1..32)")
    parser.add_argument("--min-box-side", type=float, default=32., help="Visible GT box side in ORIGINAL pixels")
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-val", action="store_true", help="Require and inspect independent val2017")
    args = parser.parse_args()
    if not 1 <= args.images <= 32 or args.min_box_side <= 0 or args.max_objects <= 0:
        parser.error("Use 1..32 images and positive minimum box side / maximum object count")
    config = read_json(args.config)
    if config.get("schema") != 1:
        raise ValueError("Expected config schema=1")
    if args.vit_path:
        config["vit_path"] = args.vit_path
    for key in ("train_images", "train_annotations", "val_images", "val_annotations"):
        if getattr(args, key):
            config["data"][key] = getattr(args, key)
        config["data"][key] = str(Path(config["data"][key]).expanduser().resolve())
    config["vit_path"] = str(Path(config["vit_path"]).expanduser().resolve())
    config["training"]["seed"] = args.seed
    paths = config["data"]
    document = read_json(paths["train_annotations"])
    subset, eligible = select_easy(document, args.images, args.min_box_side, args.max_objects, args.seed)
    # Resolve selected paths before writing; no copied images or source edits.
    for info in subset["images"]:
        image_path(paths["train_images"], info)
    val_report = {"checked": False, "note": "Fitting requires train2017 only; pilot requires independent val2017."}
    if args.check_val:
        val_document = read_json(paths["val_annotations"])
        train_images, _, categories = check_document(document)
        val_images, _, val_categories = check_document(val_document)
        if [(c["id"], c["name"]) for c in categories] != [(c["id"], c["name"]) for c in val_categories]:
            raise ValueError("Train/val category ID/name mappings differ")
        if (paths["train_annotations"] == paths["val_annotations"]
                or set(train_images) & set(val_images)
                or {im["file_name"] for im in train_images.values()} & {im["file_name"] for im in val_images.values()}):
            raise ValueError("Training and validation splits overlap")
        for info in val_document["images"]:
            image_path(paths["val_images"], info)
        val_report = {"checked": True, "images": len(val_images),
                      "annotation_sha256": sha256(paths["val_annotations"])}
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory; previous selections are preserved")
    output.mkdir(parents=True, exist_ok=True)
    subset_path = output / "overfit_annotations.json"
    write_json(subset_path, subset)
    fitting_config = deepcopy(config)
    fitting_config["data"]["train_annotations"] = str(subset_path)
    # No change to class vocabulary, architecture, query count, loss or resizing.
    write_json(output / "coco_overfit.json", fitting_config)
    write_json(output / "coco_pilot.json", config)
    save_gt(subset, paths["train_images"], output / "ground_truth")
    counts = Counter(ann["category_id"] for ann in subset["annotations"])
    report = {
        "purpose": "TRAINING_SUBSET_DEBUG_ONLY", "source_annotations": paths["train_annotations"],
        "source_annotation_sha256": sha256(paths["train_annotations"]), "seed": args.seed,
        "selection": {"images": args.images, "eligible_images": eligible,
                      "min_visible_box_side_original_px": args.min_box_side, "max_objects": args.max_objects,
                      "crowd_or_ignore_images_excluded": True, "all_selected_image_annotations_kept": True},
        "selected_image_ids": [im["id"] for im in subset["images"]],
        "objects": len(subset["annotations"]), "classifier_categories": len(subset["categories"]),
        "selected_category_counts": {c["name"]: counts[c["id"]] for c in subset["categories"] if counts[c["id"]]},
        "validation": val_report, "vit_path": config["vit_path"],
        "vit_weights_present": (Path(config["vit_path"]) / "vit.safetensors").is_file(),
        "overfit_config": str(output / "coco_overfit.json"), "pilot_config": str(output / "coco_pilot.json"),
        "note": "Easy subset excludes whole difficult images. It is intentionally biased and cannot measure generalization. "
                "Pilot config uses original complete train annotations; --limit-train samples independently of easy selection."
    }
    write_json(output / "prepare_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
