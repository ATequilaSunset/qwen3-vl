"""MOT geometry/ignore rules, sequence leakage, and actual cached training checks."""

import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch

from test_detection import detection_data, checkpoint
from vit_probe.criterion import SetCriterion
from vit_probe.data import CocoDetectionData, check_disjoint
from vit_probe.evaluate import coco_metrics
from vit_probe.prepare_visdrone import convert, discover, main, read_mot, split_groups


@pytest.fixture
def mot_data(tmp_path):
    root = tmp_path / "VisDrone2019-MOT-train"
    (root / "annotations").mkdir(parents=True)
    # Two clips from one video must stay in the same split.
    for sequence in ("uav0000001_00000_v", "uav0000001_00100_v", "uav0000002_00000_v",
                     "uav0000003_00000_v", "uav0000004_00000_v", "uav0000005_00000_v"):
        directory = root / "sequences" / sequence
        directory.mkdir(parents=True)
        rows = []
        for frame in (1, 31):
            Image.new("RGB", (96, 64), (30, frame, 100)).save(directory / f"{frame:07d}.jpg")
            rows.extend([f"{frame},7,4,8,12,20,1,4,1,2",  # score=1 car; occlusion retained
                         f"{frame},8,22,10,4,6,1,10,0,0",  # tiny motor retained
                         f"{frame},0,50,0,40,60,0,0,0,0",  # ignored region
                         f"{frame},9,50,5,10,10,0,4,0,0",  # ignored car, not positive
                         f"{frame},10,70,5,10,10,1,11,0,0"])  # others -> ignore
        (root / "annotations" / f"{sequence}.txt").write_text("\n".join(rows) + "\n")
    return root


def test_mot_columns_size_and_group_disjointness(mot_data, tmp_path):
    entries = discover(mot_data)
    train, val = split_groups(entries, .2, 42)
    assert {e["group"] for e in train}.isdisjoint({e["group"] for e in val})
    assert split_groups(entries, .2, 42) == (train, val)
    converted, audit = convert(train, 30)
    positives = [a for a in converted["annotations"] if not a["iscrowd"]]
    assert positives[0]["bbox"] == [4, 8, 12, 20]  # track ID must not become x
    assert positives[0]["track_id"] == 7 and positives[0]["occlusion"] == 2
    assert positives[1]["bbox"] == [22, 10, 4, 6]  # tiny objects never removed
    assert len(converted["categories"]) == 10
    assert len(converted["images"][0]["ignore_regions_xywh"]) == 3
    assert audit["counts"]["positive_objects"] == 2 * len(converted["images"])
    assert len(converted["annotations"]) == 32 * len(converted["images"])
    malformed = tmp_path / "det_eight_columns.txt"
    malformed.write_text("4,8,12,20,1,4,0,0\n")
    with pytest.raises(ValueError, match="10-column"):
        read_mot(malformed)


def test_ignore_loss_has_no_unmatched_background_gradient_and_gt_wins():
    criterion = SetCriterion(1)
    target = {"labels": torch.tensor([0]), "boxes": torch.tensor([[.2, .2, .1, .1]]),
              "ignore_boxes": torch.tensor([[.8, .8, .4, .4]])}
    boxes = torch.tensor([[[.2, .2, .1, .1], [.8, .8, .1, .1]]], requires_grad=True)
    logits = torch.tensor([[[0., 0.], [4., -4.]]], requires_grad=True)
    loss, _ = criterion({"pred_boxes": boxes, "pred_logits": logits}, [target])
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() == 0
    # A valid target inside ignore still supplies positive supervision.
    logits2 = torch.zeros(1, 1, 2, requires_grad=True)
    target2 = {**target, "ignore_boxes": torch.tensor([[.5, .5, 1., 1.]])}
    loss2, _ = criterion({"pred_boxes": boxes[:, :1], "pred_logits": logits2}, [target2])
    loss2.backward()
    assert logits2.grad.abs().sum() > 0
    # All queries ignored + no positives must be finite with zero gradients.
    empty = {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4),
             "ignore_boxes": target2["ignore_boxes"]}
    logits3 = torch.zeros(1, 2, 2, requires_grad=True)
    loss3, _ = criterion({"pred_boxes": boxes.detach().requires_grad_(), "pred_logits": logits3}, [empty])
    assert float(loss3.detach()) == 0 and torch.isfinite(loss3)
    loss3.backward()
    assert logits3.grad.abs().sum() == 0


def test_visdrone_preparation_crowd_metrics_and_real_training(mot_data, detection_data, tmp_path, monkeypatch):
    _, config_path, _ = detection_data
    output = tmp_path / "prepared_mot"
    argv = ["prepare", "--root", str(mot_data), "--config", str(config_path), "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    report = json.loads((output / "prepare_report.json").read_text())
    assert len(report["fitting_frames"]) == 4
    assert len({im["sequence"] for im in report["fitting_frames"]}) == 4
    assert len(list((output / "ground_truth").glob("*.png"))) == 4
    config = json.loads((output / "visdrone_pilot.json").read_text())
    train = CocoDetectionData(config["data"]["train_images"], config["data"]["train_annotations"])
    val = CocoDetectionData(config["data"]["val_images"], config["data"]["val_annotations"])
    check_disjoint(train, val)  # repeated 0000001.jpg names across videos are allowed
    assert train[0][1]["ignore_boxes"].shape == (3, 4)
    # Deliberately claim the same source video in the other split.
    original_group = val.images[0]["video_group"]
    val.images[0]["video_group"] = train.images[0]["video_group"]
    with pytest.raises(ValueError, match="video groups"):
        check_disjoint(train, val)
    val.images[0]["video_group"] = original_group
    document = json.loads((output / "overfit_annotations.json").read_text())
    fitting = CocoDetectionData(config["data"]["train_images"], output / "overfit_annotations.json")
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": .8}
               for a in document["annotations"] if not a["iscrowd"]]
    # A higher-scoring prediction inside an ignored region must not become FP.
    for info in document["images"]:
        perfect.append({"image_id": info["id"], "category_id": 4, "bbox": [55, 10, 5, 5], "score": .99})
    metrics = coco_metrics(fitting, perfect, 4)
    assert metrics["coco_maxDets_100"]["AP"] == pytest.approx(1.)
    assert "custom" in metrics["ignore_protocol"]
    # A higher-scoring prediction outside ignore must affect precision.
    bad = perfect + [{"image_id": info["id"], "category_id": 4, "bbox": [32, 35, 5, 5], "score": 1.}
                     for info in document["images"]]
    assert coco_metrics(fitting, bad, 4)["coco_maxDets_100"]["AP"] < 1.
    from vit_probe.overfit_cached import main as fit
    run = tmp_path / "mot_fit"
    monkeypatch.setattr(sys, "argv", ["fit", "--config", str(output / "visdrone_overfit.json"),
                                      "--limit-train", "4", "--steps", "2", "--batch-size", "2",
                                      "--eval-every", "2", "--device", "cpu", "--output-dir", str(run)])
    fit()
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    result = json.loads((run / "step_000002.json").read_text())
    assert result["per_image"][0]["processed_wh"]
    assert sum(v["objects"] for v in result["per_image"][0]["size_groups_diagnostic"].values()) == 2
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        main()
