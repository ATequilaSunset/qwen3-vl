"""Frozen-ViT, cached-feature fitting diagnostic. Training-subset results only.

The backbone is run once per selected image in FP32 without augmentations.
The SAME query detector is then trained in FP32 with constant learning rate,
zero dropout and an explicit optimizer-step budget. No GT enters the features.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import transformers
from transformers import AutoImageProcessor

from .criterion import SetCriterion, cxcywh_to_xyxy
from .data import CocoDetectionData, ImageCollator, file_sha256, load_config, move_batch, resolve_query_count
from .detection import build_detector
from .evaluate import coco_metrics, save_views, to_coco_predictions
from .train import fingerprint, write_json


@torch.no_grad()
def cache_features(model, dataset, processor, device):
    if not model.frozen or any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Feature caching is only valid for a frozen backbone")
    model.eval()
    collator = ImageCollator(processor)
    features, targets = [], []
    for i in range(len(dataset)):
        inputs, batch_targets = move_batch(*collator([dataset[i]]), device)
        feature = model.backbone(**inputs).feature_maps[0].detach().float()
        if not torch.isfinite(feature).all():
            raise FloatingPointError("Non-finite cached ViT features")
        features.append(feature)
        targets.extend(batch_targets)
        print(json.dumps({"cached_images": i + 1, "total_images": len(dataset),
                          "feature_shape": list(feature.shape)}), flush=True)
    return features, targets


def box_iou(boxes, truth):
    a, b = cxcywh_to_xyxy(boxes).clamp(0, 1), cxcywh_to_xyxy(truth).clamp(0, 1)
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp(min=1e-9)


@torch.no_grad()
def localization_diagnostics(output, targets, criterion):
    """Separate wrong class from wrong position; best-IoU is an optimistic diagnostic.

    Best-IoU checks all queries without a confidence threshold or one-to-one
    assignment. It is NOT detection recall or a replacement for COCO metrics.
    Hungarian matched statistics use the existing class + box matching cost.
    """
    logits, boxes = output["pred_logits"].float(), output["pred_boxes"].float()
    assignments = criterion.match(logits, boxes, targets)
    records = []
    for i, (target, (rows, cols)) in enumerate(zip(targets, assignments)):
        count = len(target["labels"])
        record = {"image_id": target["image_id"], "objects": count,
                  "mean_background_probability": float(logits[i].softmax(-1)[:, -1].mean()),
                  "best_iou_mean": None, "gt_fraction_with_any_query_iou50": None,
                  "matched_iou_mean": None, "matched_class_accuracy": None,
                  "matched_center_error_px": None}
        if count:
            iou = box_iou(boxes[i], target["boxes"])
            best = iou.max(0).values
            scale = boxes.new_tensor(target["original_wh"])
            error = ((boxes[i, rows, :2] - target["boxes"][cols, :2]) * scale).norm(dim=-1)
            record.update({"best_iou_mean": float(best.mean()),
                           "gt_fraction_with_any_query_iou50": float((best >= .5).float().mean()),
                           "matched_iou_mean": float(iou[rows, cols].mean()),
                           "matched_class_accuracy": float((logits[i, rows].argmax(-1) == target["labels"][cols]).float().mean()),
                           "matched_center_error_px": float(error.mean())})
        records.append(record)
    return records


def gradient_norms(head):
    return {name: float(sum((p.grad.detach().float().square().sum()
                            for p in component.parameters() if p.grad is not None),
                           torch.zeros((), device=next(head.parameters()).device)).sqrt())
            for name, component in head.named_children()}


@torch.no_grad()
def evaluate_cache(head, criterion, features, targets, dataset, batch_size, output_dir, step):
    head.eval()
    predictions, per_image = [], []
    totals = {key: 0. for key in ("loss", "ce", "l1", "giou")}
    for start in range(0, len(features), batch_size):
        selected_targets = targets[start:start + batch_size]
        output = head(features[start:start + batch_size])
        loss, parts = criterion(output, selected_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        size = len(selected_targets)
        totals["loss"] += float(loss) * size
        for key in ("ce", "l1", "giou"):
            totals[key] += float(parts[key]) * size
        predictions.extend(to_coco_predictions(output, selected_targets, dataset.categories))
        per_image.extend(localization_diagnostics(output, selected_targets, criterion))
    losses = {key: value / len(features) for key, value in totals.items()}
    # 'loss' includes all decoder auxiliary losses; ce/l1/giou are final-layer terms.
    metrics = coco_metrics(dataset, predictions, head.num_queries)
    record = {"step": step, "metric_scope": "TRAINING_SUBSET_DEBUG_ONLY", "losses": losses,
              "evaluation": metrics, "per_image": per_image,
              "diagnostic_note": "best-IoU uses all queries, ignores classes/scores, and permits reuse of a prediction; NOT detection recall"}
    write_json(output_dir / f"step_{step:06d}.json", record)
    write_json(output_dir / "predictions_last.json", predictions)
    save_views(dataset, predictions, output_dir / "visualizations_last")
    print(json.dumps({"step": step, **losses, **{k: metrics["coco_maxDets_100"][k]
                                               for k in ("AP", "AP50", "AR")}}), flush=True)
    return record


def save_head(path, head, optimizer, config, report, step):
    temp = Path(str(path) + ".tmp")
    torch.save({"format": "qwen_vit_cached_head_v1", "step": step,
                "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "report": report}, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-train", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.limit_train <= 32 or min(args.steps, args.batch_size, args.eval_every, args.log_every) < 1:
        parser.error("Use 1..32 images and positive step/batch/interval values")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        parser.error("Requested CPU/CUDA device is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory; previous runs are preserved")
    config = deepcopy(load_config(args.config))
    original_dropout = config["head"]["dropout"]
    config["head"]["dropout"] = 0.
    settings = config["training"]
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = config["data"]
    dataset = CocoDetectionData(data["train_images"], data["train_annotations"], limit=args.limit_train, seed=seed)
    if not sum(len(dataset.boxes[im["id"]]) for im in dataset.images):
        raise ValueError("Selected subset has no objects; choose a larger subset or another seed")
    config["head"].update(num_queries=resolve_query_count(config, dataset), num_classes=len(dataset.categories))
    settings.update(precision="float32", batch_size=args.batch_size, accumulation_steps=1,
                    lr_schedule="constant", optimizer_steps=args.steps, epochs=None)
    config["mode"] = "frozen"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "status.json", {"status": "caching_features"})
    processor = AutoImageProcessor.from_pretrained(config["vit_path"], local_files_only=True, **config["preprocessing"])
    model = build_detector(config["vit_path"], config["head"], frozen=True).float().to(device)
    head_hash = fingerprint(model.head)
    features, targets = cache_features(model, dataset, processor, device)
    head = model.head
    assert all(p.grad is None for p in model.backbone.parameters())
    # No graph connects cached features to the backbone. Release its weights.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    criterion = SetCriterion(len(dataset.categories)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["head_lr"], weight_decay=settings["weight_decay"])
    report = {"config": config, "dataset": dataset.summary(), "head_initial_sha256": head_hash,
              "vit_weights_sha256": file_sha256(Path(config["vit_path"]) / "vit.safetensors"),
              "feature_shapes": [list(f.shape) for f in features],
              "cache_bytes": sum(f.numel() * f.element_size() for f in features),
              "torch": str(torch.__version__), "transformers": transformers.__version__,
              "device": str(device), "metric_scope": "TRAINING_SUBSET_DEBUG_ONLY",
              "changes_from_regular_training": {"original_dropout": original_dropout, "dropout": 0.,
                                                 "constant_lr": settings["head_lr"], "precision": "float32",
                                                 "cache_frozen_features": True, "optimizer_steps": args.steps},
              "checkpoint_note": "head-only diagnostic checkpoint; not compatible with vit_probe.evaluate CLI"}
    write_json(output_dir / "run_config.json", report)
    evaluate_cache(head, criterion, features, targets, dataset, args.batch_size, output_dir, 0)
    rng = random.Random(seed)
    order, cursor = [], 0
    best_ap, start = -1., time.monotonic()
    gradient_check_step = None
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order = list(range(len(features)))
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor:cursor + args.batch_size]
        cursor += len(indices)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        output = head([features[i] for i in indices])
        batch_targets = [targets[i] for i in indices]
        batch_objects = sum(len(target["labels"]) for target in batch_targets)
        loss, parts = criterion(output, batch_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        check_gradients = gradient_check_step is None and batch_objects > 0
        log = step == 1 or check_gradients or step % args.log_every == 0 or step == args.steps
        norms = gradient_norms(head) if log else None
        # An empty image legitimately supplies no box regression gradient.
        if check_gradients:
            if not all(norms[name] > 0 for name in ("projection", "classifier", "box_head")):
                raise RuntimeError(f"Missing gradient in a required detector component: {norms}")
            gradient_check_step = step
        total_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        if log:
            item = {"step": step, "batch_objects": batch_objects, "loss": float(loss.detach()),
                    **{key: float(parts[key].detach()) for key in ("ce", "l1", "giou")},
                    "lr": optimizer.param_groups[0]["lr"], "grad_norm_before_clip": float(total_norm),
                    "component_grad_norms": norms, "seconds": time.monotonic() - start}
            with (output_dir / "train_steps.jsonl").open("a") as stream:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
            print(json.dumps(item, allow_nan=False), flush=True)
            write_json(output_dir / "status.json", {"status": "running", "completed_step": step, "total_steps": args.steps})
        if step % args.eval_every == 0 or step == args.steps:
            record = evaluate_cache(head, criterion, features, targets, dataset, args.batch_size, output_dir, step)
            save_head(output_dir / "head_last.pt", head, optimizer, config, report, step)
            ap = record["evaluation"]["coco_maxDets_100"]["AP"]
            if ap is not None and ap > best_ap:
                best_ap = ap
                import shutil
                shutil.copyfile(output_dir / "head_last.pt", output_dir / "head_best.pt.tmp")
                (output_dir / "head_best.pt.tmp").replace(output_dir / "head_best.pt")
    write_json(output_dir / "status.json", {"status": "completed", "steps": args.steps, "best_AP": best_ap,
                                            "gradient_check_step": gradient_check_step,
                                            "seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
