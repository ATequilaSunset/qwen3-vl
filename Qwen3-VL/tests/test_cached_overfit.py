"""Cache equivalence, class/position separation and real tiny-ViT CLI tests."""

import json
import sys

import pytest
import torch
from test_detection import detection_data, checkpoint
from vit_probe.criterion import SetCriterion
from vit_probe.data import CocoDetectionData, ImageCollator, load_config
from vit_probe.detection import build_detector
from vit_probe.overfit_cached import cache_features, localization_diagnostics, main
from vit_probe.train import fingerprint


def test_cached_feature_detector_equals_online_frozen_detector(detection_data):
    splits, config_path, processor = detection_data
    config = load_config(config_path)
    dataset = CocoDetectionData(*splits["train"])
    model = build_detector(config["vit_path"], dict(config["head"], num_classes=2), frozen=True).eval()
    original_hash = fingerprint(model.backbone)
    features, targets = cache_features(model, dataset, processor, torch.device("cpu"))
    assert all(not f.requires_grad and f.dtype == torch.float32 for f in features)
    with torch.no_grad():
        inputs, _ = ImageCollator(processor)([dataset[0], dataset[1]])
        online = model(**inputs)
        cached = model.head(features)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(cached[key], online[key], atol=1e-5, rtol=1e-5)
    before = fingerprint(model.head)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001)
    loss, _ = SetCriterion(2)(model.head(features), targets)
    loss.backward()
    optimizer.step()
    assert fingerprint(model.head) != before
    assert fingerprint(model.backbone) == original_hash
    assert all(p.grad is None for p in model.backbone.parameters())


def test_localization_diagnostics_distinguish_wrong_class_from_wrong_box():
    target = {"image_id": 5, "original_wh": (800, 800), "labels": torch.tensor([0]),
              "boxes": torch.tensor([[.9, .9, .01, .01]])}
    output = {"pred_logits": torch.tensor([[[-10., 10., -10.]]]),
              "pred_boxes": target["boxes"][None]}
    record = localization_diagnostics(output, [target], SetCriterion(2))[0]
    assert record["best_iou_mean"] == pytest.approx(1.)
    assert record["gt_fraction_with_any_query_iou50"] == 1.
    assert record["matched_class_accuracy"] == 0.
    assert record["matched_center_error_px"] == 0.
    output["pred_boxes"] = torch.tensor([[[.5, .5, .01, .01]]])
    record = localization_diagnostics(output, [target], SetCriterion(2))[0]
    assert record["best_iou_mean"] == 0.
    assert record["matched_center_error_px"] > 400
    empty = {**target, "labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}
    assert localization_diagnostics(output, [empty], SetCriterion(2))[0]["best_iou_mean"] is None


def test_real_cached_cli_budget_metrics_and_checkpoints(detection_data, tmp_path, monkeypatch):
    _, config, _ = detection_data
    output = tmp_path / "cached"
    monkeypatch.setattr(sys, "argv", ["cached", "--config", str(config), "--device", "cpu",
                                      "--limit-train", "2", "--steps", "5", "--batch-size", "1",
                                      "--eval-every", "3", "--log-every", "1", "--output-dir", str(output)])
    main()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed" and status["steps"] == 5
    report = json.loads((output / "run_config.json").read_text())
    assert report["config"]["head"]["dropout"] == 0
    assert report["config"]["training"]["precision"] == "float32"
    assert report["metric_scope"] == "TRAINING_SUBSET_DEBUG_ONLY"
    logs = [json.loads(line) for line in (output / "train_steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in logs] == [1, 2, 3, 4, 5]
    assert len({row["lr"] for row in logs}) == 1
    assert logs[0]["batch_objects"] == 0  # seed 42 selects the empty image first
    assert logs[0]["component_grad_norms"]["box_head"] == 0
    assert logs[1]["component_grad_norms"]["box_head"] > 0
    assert status["gradient_check_step"] == 2
    assert [p.name for p in sorted(output.glob("step_*.json"))] == ["step_000000.json", "step_000003.json", "step_000005.json"]
    final = json.loads((output / "step_000005.json").read_text())
    assert len(final["per_image"]) == 2
    assert set(final["losses"]) == {"loss", "ce", "l1", "giou"}
    state = torch.load(output / "head_last.pt", weights_only=True)
    assert state["step"] == 5
    assert state["format"] == "qwen_vit_cached_head_v1"
    assert not any("backbone" in key for key in state["head"])
    with pytest.raises(FileExistsError):
        main()
