"""Usage: python -m vit_probe.extract --model-path PATH --output-dir PATH."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen3-VL ViT without mergers or LLM")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32",
                        help="float32 avoids an additional lossy conversion during extraction")
    args = parser.parse_args()

    from transformers import AutoImageProcessor
    from .backbone import DTYPES, PreMergerViT, check_version

    check_version()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("--output-dir must be absent or empty; existing files are not overwritten")
    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = PreMergerViT.from_qwen_checkpoint(args.model_path, dtype=DTYPES[args.dtype])
    for processor_key, config_key in (("patch_size", "patch_size"),
                                      ("temporal_patch_size", "temporal_patch_size"),
                                      ("merge_size", "spatial_merge_size")):
        if getattr(processor, processor_key, None) != getattr(model.vision.config, config_key):
            raise ValueError(f"Processor/model mismatch: {processor_key}; use the matching processor files")
    metadata = model.save_export(args.output_dir, processor)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved ViT to: {args.output_dir.resolve()}")
    print("This is an extraction, not a fine-tuned model. Run vit_probe.verify next.")


if __name__ == "__main__":
    main()
