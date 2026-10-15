"""Strict COCO-format image/box loading; resize-only geometry for this diagnostic."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CocoDetectionData(Dataset):
    def __init__(self, image_root, annotations, categories=None, limit=0, seed=42):
        self.image_root, self.annotation_path = Path(image_root).resolve(), Path(annotations).resolve()
        document = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.categories = sorted(document["categories"], key=lambda item: item["id"])
        actual = [(c["id"], c["name"]) for c in self.categories]
        if len({c[0] for c in actual}) != len(actual) or not actual:
            raise ValueError("Categories must have unique IDs and must not be empty")
        if categories is not None and actual != [(c["id"], c["name"]) for c in categories]:
            raise ValueError("Train/eval category ID/name mappings differ; check annotation versions")
        self.category_to_label = {c["id"]: i for i, c in enumerate(self.categories)}
        images = sorted(document["images"], key=lambda item: item["id"])
        if not images or len({im["id"] for im in images}) != len(images):
            raise ValueError("Images must have unique IDs and must not be empty")
        by_id = {im["id"]: im for im in images}
        self.boxes = {im["id"]: [] for im in images}
        self.skipped = Counter()
        annotation_ids = set()
        for annotation in document["annotations"]:
            if annotation["id"] in annotation_ids:
                raise ValueError("Duplicate annotation ID")
            annotation_ids.add(annotation["id"])
            image_id, category_id = annotation["image_id"], annotation["category_id"]
            if image_id not in by_id or category_id not in self.category_to_label:
                raise ValueError("Annotation references an unknown image or category")
            if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
                self.skipped["crowd_or_ignore"] += 1
                continue
            x, y, w, h = annotation["bbox"]
            image_w, image_h = by_id[image_id]["width"], by_id[image_id]["height"]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h, image_w, image_h) <= 0:
                raise ValueError(f"Invalid box/dimensions in annotation {annotation['id']}")
            x0, y0, x1, y1 = max(0., x), max(0., y), min(image_w, x + w), min(image_h, y + h)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"Box outside image in annotation {annotation['id']}")
            if (x0, y0, x1, y1) != (x, y, x + w, y + h):
                self.skipped["clipped_boxes"] += 1
            normalized = [(x0 + x1) / (2 * image_w), (y0 + y1) / (2 * image_h),
                          (x1 - x0) / image_w, (y1 - y0) / image_h]
            self.boxes[image_id].append((self.category_to_label[category_id], normalized))
        if limit:
            images = sorted(random.Random(seed).sample(images, min(limit, len(images))), key=lambda im: im["id"])
        self.images = images
        self.paths = []
        for im in images:
            if min(im["width"], im["height"]) <= 0:
                raise ValueError(f"Invalid image dimensions: {im['id']}")
            path = (self.image_root / im["file_name"]).resolve()
            if not path.is_relative_to(self.image_root) or not path.is_file():
                raise FileNotFoundError(f"Image missing or outside its split root: {path}")
            self.paths.append(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.paths[index]) as raw:
            if raw.size != (info["width"], info["height"]):
                raise ValueError(f"Image size disagrees with annotations: {self.paths[index]}")
            image = raw.convert("RGB")
        objects = self.boxes[info["id"]]
        target = {
            "image_id": info["id"], "original_wh": (info["width"], info["height"]),
            "labels": torch.tensor([o[0] for o in objects], dtype=torch.long),
            "boxes": torch.tensor([o[1] for o in objects], dtype=torch.float32).reshape(-1, 4),
        }
        return image, target

    def summary(self):
        counts = [len(self.boxes[im["id"]]) for im in self.images]
        return {"images": len(self), "objects": sum(counts), "max_objects_per_image": max(counts),
                "empty_images": counts.count(0), "categories": self.categories,
                "annotation_sha256": file_sha256(self.annotation_path),
                "image_root": str(self.image_root), "annotations": str(self.annotation_path),
                "selected_image_ids": [im["id"] for im in self.images],
                "annotation_adjustments_full_file": dict(self.skipped)}


class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, targets = zip(*batch)
        inputs = self.processor(images=list(images), return_tensors="pt")
        # No crop, flip or letterboxing: normalized boxes stay invariant under
        # independent x/y resize. Map back with ORIGINAL width/height at eval.
        return {key: inputs[key] for key in ("pixel_values", "image_grid_thw")}, list(targets)


def move_batch(inputs, targets, device):
    return ({key: value.to(device) for key, value in inputs.items()},
            [{key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in target.items()} for target in targets])


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("Expected detection config schema=1")
    return config


def check_disjoint(train, val):
    if train.annotation_path == val.annotation_path or set(train.paths) & set(val.paths):
        raise ValueError("Training and validation splits overlap")
    # AI-TOD filenames identify images even when trainval contains copies.
    if {p.name for p in train.paths} & {p.name for p in val.paths}:
        raise ValueError("Train/val contain the same image filename; inspect possible data leakage")


def resolve_query_count(config, dataset):
    maximum = max(len(dataset.boxes[im["id"]]) for im in dataset.images)
    requested = config["head"]["num_queries"]
    if requested == 0:
        requested = max(300, math.ceil(maximum / 50) * 50)
    if requested < maximum:
        raise ValueError(f"num_queries={requested} < max train objects={maximum}. "
                         "Increase queries or use 0 for train-only automatic selection.")
    return requested
