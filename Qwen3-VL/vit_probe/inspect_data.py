"""CPU-only preflight for split paths, category mappings and query capacity."""
import argparse
import json
from pathlib import Path

from .data import CocoDetectionData, check_disjoint, load_config, resolve_query_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="vit_probe/artifacts/data_audit.json")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    train = CocoDetectionData(data["train_images"], data["train_annotations"])
    val = CocoDetectionData(data["val_images"], data["val_annotations"], categories=train.categories)
    check_disjoint(train, val)
    # Read representative files in addition to checking ALL declared paths.
    for dataset in (train, val):
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            dataset[index]
    report = {"train": train.summary(), "val": val.summary(),
              "resolved_num_queries_from_train": resolve_query_count(config, train),
              "note": "COCO format checked. Dataset release/version requires provenance; filenames alone do not prove v2."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in ("train", "val"):
        report[split].pop("selected_image_ids")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
