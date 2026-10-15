"""Single-GPU frozen/finetuned ViT detection with identical head initialization.

The 5-epoch default is a pilot. It is not a convergence or representation-loss
claim. Start with --overfit to validate the pipeline on a tiny training subset.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from .criterion import SetCriterion
from .data import (CocoDetectionData, ImageCollator, check_disjoint, file_sha256,
                   load_config, move_batch, resolve_query_count)
from .detection import build_detector
from .evaluate import evaluate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def fingerprint(module):
    checksum = hashlib.sha256()
    for name, value in module.state_dict().items():
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def save_checkpoint(path, model, optimizer, scheduler, config, categories, epoch, best_ap, signature):
    temp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "config": config, "categories": categories,
                "epoch": epoch, "best_ap": best_ap, "signature": signature,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temp)
    temp.replace(path)


def train_epoch(model, criterion, loader, optimizer, device, precision, accumulation, clip, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, count = 0., 0
    gradient_check = None
    start = time.monotonic()
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = move_batch(inputs, targets, device)
        group_start = (step // accumulation) * accumulation
        group_length = min(accumulation, len(loader) - group_start)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            outputs = model(**inputs)
        loss, components = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
        (loss / group_length).backward()
        if gradient_check is None:
            backbone_grad = any(p.grad is not None and bool(p.grad.abs().max() > 0)
                                for p in model.backbone.parameters())
            head_grad = any(p.grad is not None and bool(p.grad.abs().max() > 0)
                            for p in model.head.parameters())
            if not head_grad or (not model.frozen and not backbone_grad):
                raise RuntimeError("Expected trainable head/backbone gradient is missing")
            if model.frozen and any(p.grad is not None for p in model.backbone.parameters()):
                raise RuntimeError("Frozen backbone unexpectedly has gradients")
            gradient_check = {"head_nonzero_grad": head_grad, "backbone_nonzero_grad": backbone_grad}
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * len(targets)
        count += len(targets)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(json.dumps({"epoch": epoch, "batch": step + 1, "batches": len(loader),
                              "loss": float(loss.detach()),
                              **{k: float(v.detach()) for k, v in components.items() if "aux" not in k}}), flush=True)
    return {"loss": total_loss / count, "seconds": time.monotonic() - start,
            "gradient_check": gradient_check,
            "peak_gpu_memory_GiB": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("frozen", "finetune"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--overfit", action="store_true", help="Evaluate SAME limited training subset; debug only")
    parser.add_argument("--resume", help="Our own last.pt checkpoint; resumes at next epoch")
    args = parser.parse_args()
    config = deepcopy(load_config(args.config))
    settings = config["training"]
    for key in ("epochs", "batch_size", "accumulation_steps", "workers", "precision"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if min(settings["epochs"], settings["batch_size"], settings["accumulation_steps"]) <= 0:
        parser.error("epochs, batch-size and accumulation-steps must be positive")
    if min(settings["workers"], args.limit_train, args.limit_val) < 0:
        parser.error("workers and subset limits must be nonnegative")
    if args.overfit and not 0 < args.limit_train <= 32:
        parser.error("--overfit requires --limit-train between 1 and 32")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        parser.error("Requested CPU/CUDA device is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if settings["precision"] == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("Use --precision float32 for CPU/unsupported GPU")
    # Seed BEFORE creating either mode's head. Data order is reseeded per epoch.
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = config["data"]
    train = CocoDetectionData(paths["train_images"], paths["train_annotations"],
                              limit=args.limit_train, seed=settings["seed"])
    val = train if args.overfit else CocoDetectionData(
        paths["val_images"], paths["val_annotations"], categories=train.categories,
        limit=args.limit_val, seed=settings["seed"])
    if not args.overfit:
        check_disjoint(train, val)
    config["head"]["num_queries"] = resolve_query_count(config, train)
    config["head"]["num_classes"] = len(train.categories)
    config["mode"] = args.mode
    config["subset"] = {"limit_train": args.limit_train, "limit_val": args.limit_val, "overfit": args.overfit}
    provenance = {"train": train.summary(), "evaluation": val.summary(),
                  "vit_weights_sha256": file_sha256(Path(config["vit_path"]) / "vit.safetensors"),
                  "metric_scope": "TRAINING_SUBSET_DEBUG_ONLY" if args.overfit else "validation"}
    signature = hashlib.sha256(json.dumps({"config": config, "data": provenance}, sort_keys=True).encode()).hexdigest()
    output_dir = Path(args.output_dir)
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty; use --resume or a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(config["vit_path"], local_files_only=True,
                                                   **config["preprocessing"])
    collator = ImageCollator(processor)
    val_loader = DataLoader(val, batch_size=settings["batch_size"], num_workers=settings["workers"],
                            collate_fn=collator)
    model = build_detector(config["vit_path"], config["head"], frozen=args.mode == "frozen",
                           gradient_checkpointing=settings["gradient_checkpointing"])
    head_hash = fingerprint(model.head)
    model.to(device)
    criterion = SetCriterion(len(train.categories)).to(device)
    groups = [{"params": model.head.parameters(), "lr": settings["head_lr"]}]
    if args.mode == "finetune":
        groups.append({"params": model.backbone.parameters(), "lr": settings["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    start_epoch, best_ap = 1, -1.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume configuration/data differs. Do not mix experiments or change total epochs.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch, best_ap = checkpoint["epoch"] + 1, checkpoint["best_ap"]
        del checkpoint
    report = {"config": config, "data": provenance, "head_initial_sha256": head_hash,
              "torch": str(torch.__version__), "transformers": transformers.__version__, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
              "effective_batch_size": settings["batch_size"] * settings["accumulation_steps"],
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "signature": signature}
    write_json(output_dir / "run_config.json", report)
    write_json(output_dir / "status.json", {"status": "running", "next_epoch": start_epoch})
    print(json.dumps({"mode": args.mode, "queries": model.head.num_queries, "train_images": len(train),
                      "eval_images": len(val), "metric_scope": provenance["metric_scope"],
                      "head_initial_sha256": head_hash, "effective_batch_size": report["effective_batch_size"]}), flush=True)
    for epoch in range(start_epoch, settings["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator().manual_seed(settings["seed"] + epoch)
        loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, generator=generator,
                            num_workers=settings["workers"], collate_fn=collator)
        losses = train_epoch(model, criterion, loader, optimizer, device, settings["precision"],
                             settings["accumulation_steps"], settings["clip_grad_norm"], epoch)
        scheduler.step()
        metrics = evaluate(model, val_loader, val, device, settings["precision"], output_dir / "predictions_last.json")
        record = {"epoch": epoch, "train": losses, "evaluation": metrics,
                  "metric_scope": provenance["metric_scope"]}
        write_json(output_dir / f"epoch_{epoch:03d}.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)
        ap = metrics["coco_maxDets_100"]["AP"]
        improved = ap is not None and ap > best_ap
        if improved:
            best_ap = ap
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, config,
                        train.categories, epoch, best_ap, signature)
        if improved:
            # Copy the completed atomic checkpoint; do not serialize GPU state twice.
            import shutil
            shutil.copyfile(output_dir / "last.pt", output_dir / "best.pt.tmp")
            (output_dir / "best.pt.tmp").replace(output_dir / "best.pt")
        write_json(output_dir / "status.json", {"status": "running", "completed_epoch": epoch, "best_AP": best_ap})
    write_json(output_dir / "status.json", {"status": "completed", "epochs": settings["epochs"], "best_AP": best_ap})


if __name__ == "__main__":
    main()
