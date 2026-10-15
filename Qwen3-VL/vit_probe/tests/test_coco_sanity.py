"""COCO control: preserve all GT/80 sparse classes, exclude leakage, exercise CLI."""

from copy import deepcopy
import json
import sys

from PIL import Image
import pytest

from test_detection import detection_data, checkpoint
from vit_probe.data import CocoDetectionData
from vit_probe.evaluate import coco_metrics
from vit_probe.prepare_coco import main, select_easy


def coco_document(offset=0):
    document = {"images": [], "annotations": [],
                "categories": [{"id": i, "name": f"class_{i}"} for i in list(range(1, 80)) + [90]]}
    for i in range(1, 8):
        document["images"].append({"id": offset + i, "file_name": f"image_{offset + i}.png",
                                   "width": 128, "height": 96})
        if i == 7:  # Empty image must not enter easy fitting.
            continue
        document["annotations"].append({"id": offset + i, "image_id": offset + i,
                                         "category_id": 90 if i == 4 else 1,
                                         "bbox": [10, 12, 40, 40], "area": 1600,
                                         "iscrowd": int(i == 6)})
    # A mixed large+small scene is excluded as a whole, not relabeled partially.
    document["annotations"].append({"id": offset + 8, "image_id": offset + 5,
                                     "category_id": 1, "bbox": [60, 50, 4, 4], "area": 16, "iscrowd": 0})
    # An eligible multi-object image must retain both annotations.
    document["annotations"].append({"id": offset + 9, "image_id": offset + 2,
                                     "category_id": 90, "bbox": [70, 20, 40, 40], "area": 1600, "iscrowd": 0})
    return document


def test_easy_selection_keeps_all_annotations_and_sparse_categories():
    document = coco_document()
    original = deepcopy(document)
    subset, eligible = select_easy(document, 4, 32, 10, 42)
    assert document == original
    assert eligible == 4 and [im["id"] for im in subset["images"]] == [1, 2, 3, 4]
    assert subset["categories"] == document["categories"] and len(subset["categories"]) == 80
    assert subset["annotations"] == [ann for ann in document["annotations"] if ann["image_id"] in {1, 2, 3, 4}]
    assert len(subset["annotations"]) == 5
    assert select_easy(document, 2, 32, 10, 42) == select_easy(document, 2, 32, 10, 42)
    with pytest.raises(ValueError, match="eligible"):
        select_easy(document, 5, 32, 10, 42)
    malformed = deepcopy(document)
    malformed["categories"] = malformed["categories"][:-1]
    with pytest.raises(ValueError, match="80-category"):
        select_easy(malformed, 4, 32, 10, 42)


def test_coco_prepare_oracle_metrics_and_cached_cli(detection_data, tmp_path, monkeypatch):
    _, base_path, _ = detection_data
    config = json.loads(base_path.read_text())
    for split, offset in (("train", 0), ("val", 1000)):
        root = tmp_path / f"coco_{split}"
        root.mkdir()
        document = coco_document(offset)
        for info in document["images"]:
            Image.new("RGB", (128, 96), (info["id"] % 255, 20, 80)).save(root / info["file_name"])
        annotations = tmp_path / f"instances_{split}2017.json"
        annotations.write_text(json.dumps(document))
        config["data"][f"{split}_images"] = str(root)
        config["data"][f"{split}_annotations"] = str(annotations)
    base_path.write_text(json.dumps(config))
    output = tmp_path / "coco_control"
    argv = ["prepare", "--config", str(base_path), "--images", "4", "--check-val", "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    report = json.loads((output / "prepare_report.json").read_text())
    assert report["validation"]["checked"] and report["classifier_categories"] == 80
    assert len(list((output / "ground_truth").glob("*.png"))) == 4
    overfit = json.loads((output / "coco_overfit.json").read_text())
    pilot = json.loads((output / "coco_pilot.json").read_text())
    assert pilot["data"]["train_annotations"] == config["data"]["train_annotations"]
    assert pilot["head"] == config["head"] == overfit["head"]
    data = CocoDetectionData(overfit["data"]["train_images"], overfit["data"]["train_annotations"])
    assert data.category_to_label[90] == 79
    doc = json.loads((output / "overfit_annotations.json").read_text())
    predictions = [{"image_id": a["image_id"], "category_id": a["category_id"],
                    "bbox": a["bbox"], "score": 1.} for a in doc["annotations"]]
    assert coco_metrics(data, predictions, 4)["coco_maxDets_100"]["AP"] == pytest.approx(1.)
    with pytest.raises(FileExistsError):
        main()
    # The full class count reaches the actual cached-feature training CLI.
    from vit_probe.overfit_cached import main as fit
    run = tmp_path / "coco_fit"
    monkeypatch.setattr(sys, "argv", ["fit", "--config", str(output / "coco_overfit.json"),
                                      "--device", "cpu", "--limit-train", "4", "--steps", "2",
                                      "--batch-size", "2", "--eval-every", "2", "--output-dir", str(run)])
    fit()
    run_report = json.loads((run / "run_config.json").read_text())
    assert run_report["config"]["head"]["num_classes"] == 80
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    # Selecting train itself as validation must fail before writing output.
    monkeypatch.setattr(sys, "argv", argv + ["--val-images", config["data"]["train_images"],
                                             "--val-annotations", config["data"]["train_annotations"],
                                             "--output-dir", str(tmp_path / "leaked")])
    with pytest.raises(ValueError, match="overlap"):
        main()
    assert not (tmp_path / "leaked").exists()


def test_missing_annotations_stop_before_preparation(tmp_path, monkeypatch):
    config = {"schema": 1, "vit_path": str(tmp_path / "vit"), "training": {},
              "data": {f"{s}_{kind}": str(tmp_path / f"{s}_{kind}")
                       for s in ("train", "val") for kind in ("images", "annotations")}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(path), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(FileNotFoundError, match="instances JSON"):
        main()
    assert not (tmp_path / "out").exists()
