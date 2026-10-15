"""Meaningful geometry, assignment, masking, gradient and real tiny-ViT CLI checks."""

import json
from pathlib import Path
import sys

import pytest
import torch

from test_backbone import checkpoint  # reuse the tiny REAL upstream Qwen checkpoint fixture
from vit_probe.backbone import PreMergerViT
from vit_probe.criterion import SetCriterion, cxcywh_to_xyxy, generalized_iou
from vit_probe.data import CocoDetectionData, ImageCollator, resolve_query_count
from vit_probe.detection import DetectionHead
from vit_probe.evaluate import coco_metrics, to_coco_predictions


@pytest.fixture
def detection_data(checkpoint, tmp_path):
    path, _, processor, images, _ = checkpoint
    export = tmp_path / "export"
    PreMergerViT.from_qwen_checkpoint(path).save_export(export, processor)
    categories = [{"id": 3, "name": "ship"}, {"id": 9, "name": "vehicle"}]
    splits = {}
    for split in ("train", "val"):
        root = tmp_path / split
        root.mkdir()
        document = {"images": [], "annotations": [], "categories": categories}
        for i, image in enumerate(images):
            name = f"{split}_{i}.png"
            image.save(root / name)
            document["images"].append({"id": i + 1, "file_name": name,
                                       "width": image.width, "height": image.height})
            if i == 0:  # second image intentionally has no objects
                document["annotations"].append({"id": 1, "image_id": i + 1, "category_id": 3,
                                                "bbox": [2, 4, 4, 6], "area": 24, "iscrowd": 0})
        annotation = tmp_path / f"{split}.json"
        annotation.write_text(json.dumps(document))
        splits[split] = (root, annotation)
    config = {"schema": 1, "vit_path": str(export),
              "data": {f"{s}_{kind}": str(splits[s][i]) for s in splits
                       for i, kind in enumerate(("images", "annotations"))},
              "head": {"num_queries": 4, "hidden_dim": 32, "decoder_layers": 2,
                       "heads": 4, "ffn_dim": 64, "dropout": 0.},
              "training": {"epochs": 1, "batch_size": 2, "accumulation_steps": 2,
                           "head_lr": .001, "backbone_lr": .0001, "weight_decay": .0001,
                           "clip_grad_norm": 1., "seed": 42, "workers": 0,
                           "precision": "float32", "gradient_checkpointing": True},
              "preprocessing": {"min_pixels": 64, "max_pixels": 1024}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return splits, config_path, processor


def test_box_roundtrip_categories_resize_and_empty_targets(detection_data):
    splits, _, processor = detection_data
    data = CocoDetectionData(*splits["train"])
    image, target = data[0]
    torch.testing.assert_close(target["boxes"], torch.tensor([[4 / 24, 7 / 16, 4 / 24, 6 / 16]]))
    assert target["labels"].tolist() == [0]  # COCO ID 3 -> contiguous class 0
    assert data[1][1]["boxes"].shape == (0, 4)
    inputs, targets = ImageCollator(processor)([data[0], data[1]])
    assert inputs["image_grid_thw"].tolist() == [[1, 4, 6], [1, 8, 4]]
    predictions = to_coco_predictions(
        {"pred_logits": torch.tensor([[[10., -10., -10.]]]),
         "pred_boxes": target["boxes"][None]}, [target], data.categories)
    assert predictions[0]["category_id"] == 3
    assert predictions[0]["bbox"] == pytest.approx([2, 4, 4, 6])
    assert resolve_query_count({"head": {"num_queries": 0}}, data) == 300
    wrong = [{"id": 3, "name": "other"}, {"id": 9, "name": "vehicle"}]
    with pytest.raises(ValueError, match="mappings differ"):
        CocoDetectionData(*splits["val"], categories=wrong)


def test_assignment_permutation_background_and_capacity():
    torch.set_num_threads(1)
    criterion = SetCriterion(2)
    targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.tensor([[.2, .3, .1, .1], [.8, .7, .2, .2]])},
               {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
    boxes = torch.tensor([[[.8, .7, .2, .2], [.2, .3, .1, .1]], [[.5, .5, .2, .2], [.2, .2, .1, .1]]],
                         requires_grad=True)
    logits = torch.tensor([[[-8., 8., -8.], [8., -8., -8.]], [[-8., -8., 8.], [-8., -8., 8.]]],
                          requires_grad=True)
    loss, parts = criterion({"pred_logits": logits, "pred_boxes": boxes}, targets)
    assert float(loss.detach()) < 1e-5
    assert float(parts["l1"].detach()) == 0
    loss.backward()
    assert torch.isfinite(boxes.grad).all() and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="More ground-truth"):
        criterion({"pred_logits": logits[:, :1], "pred_boxes": boxes[:, :1]}, targets)


def test_padding_does_not_change_first_image_output():
    torch.manual_seed(2)
    torch.set_num_threads(1)
    head = DetectionHead(16, 2, num_queries=4, hidden_dim=32, decoder_layers=2,
                         heads=4, ffn_dim=64, dropout=0).eval()
    a, b = torch.randn(16, 2, 3), torch.randn(16, 5, 4)
    with torch.no_grad():
        alone, batched = head([a]), head([a, b])
    torch.testing.assert_close(alone["pred_logits"][0], batched["pred_logits"][0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(alone["pred_boxes"][0], batched["pred_boxes"][0], atol=1e-6, rtol=1e-5)


def test_metrics_perfect_and_empty_predictions(detection_data):
    splits, _, _ = detection_data
    data = CocoDetectionData(*splits["val"])
    perfect = [{"image_id": 1, "category_id": 3, "bbox": [2, 4, 4, 6], "score": .99}]
    scores = coco_metrics(data, perfect, 300)
    assert scores["coco_maxDets_100"]["AP"] == pytest.approx(1)
    assert scores["coco_maxDets_300"]["AP"] == pytest.approx(1)
    assert scores["coco_maxDets_100"]["per_class_AP"]["vehicle"] is None
    assert coco_metrics(data, [], 300)["coco_maxDets_100"]["AP"] == 0


def test_small_head_can_overfit_a_box():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    head = DetectionHead(16, 1, num_queries=4, hidden_dim=32, decoder_layers=2,
                         heads=4, ffn_dim=64, dropout=0)
    features = [torch.randn(16, 3, 4)]
    target = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[.3, .4, .15, .2]])}]
    criterion = SetCriterion(1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    initial = None
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        output = head(features)
        loss, _ = criterion(output, target)
        initial = float(loss.detach()) if initial is None else initial
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        optimizer.step()
    with torch.no_grad():
        final_output = head(features)
        final, _ = criterion(final_output, target)
        chosen = final_output["pred_logits"][0].softmax(-1)[:, 0].argmax()
        giou = generalized_iou(cxcywh_to_xyxy(final_output["pred_boxes"][0, chosen:chosen + 1]),
                               cxcywh_to_xyxy(target[0]["boxes"]))
    assert float(final) < initial * .2
    assert float(giou) > .5


def test_real_tiny_vit_training_modes_checkpoint_and_eval_cli(detection_data, tmp_path, monkeypatch):
    from vit_probe import train, evaluate, inspect_data
    splits, config_path, _ = detection_data
    monkeypatch.setattr(sys, "argv", ["inspect", "--config", str(config_path), "--output", str(tmp_path / "audit.json")])
    inspect_data.main()
    hashes = []
    for mode in ("frozen", "finetune"):
        output = tmp_path / mode
        argv = ["train", "--config", str(config_path), "--mode", mode, "--device", "cpu",
                "--output-dir", str(output)]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
        report = json.loads((output / "run_config.json").read_text())
        hashes.append(report["head_initial_sha256"])
        epoch = json.loads((output / "epoch_001.json").read_text())
        assert epoch["train"]["gradient_check"]["backbone_nonzero_grad"] == (mode == "finetune")
        checkpoint = torch.load(output / "last.pt", weights_only=True)
        assert checkpoint["epoch"] == 1
        assert json.loads((output / "status.json").read_text())["status"] == "completed"
        # Check exact-config resume loading; finished run performs zero extra epochs.
        monkeypatch.setattr(sys, "argv", argv + ["--resume", str(output / "last.pt")])
        train.main()
    assert hashes[0] == hashes[1]
    eval_path = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(tmp_path / "frozen" / "best.pt"),
                                      "--images", str(splits["val"][0]), "--annotations", str(splits["val"][1]),
                                      "--output-dir", str(eval_path), "--device", "cpu", "--precision", "float32"])
    evaluate.main()
    assert (eval_path / "metrics.json").exists()
