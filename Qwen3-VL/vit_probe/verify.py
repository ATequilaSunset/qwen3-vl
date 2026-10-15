"""Compare an exported ViT with a hook at the original main merger's INPUT."""

import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Original Qwen checkpoint")
    parser.add_argument("--vit-path", type=Path, required=True, help="Directory written by vit_probe.extract")
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON verification report")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; no automatic GPU selection")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if not 0 < args.min_pixels <= args.max_pixels or args.atol < 0 or args.rtol < 0:
        parser.error("Require 0 < min-pixels <= max-pixels and nonnegative tolerances")
    # A failed rerun must not leave a previous successful report as current evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": False, "status": "incomplete"}) + "\n", encoding="utf-8")

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoImageProcessor
    from .backbone import DTYPES, PreMergerViT, check_version, load_original_vision

    check_version()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Supported devices: cpu or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or check the server environment")
    if device.type == "cpu" and args.dtype != "float32":
        parser.error("Use float32 for the CPU comparison")
    dtype = DTYPES[args.dtype]
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    processor_kwargs = dict(local_files_only=True, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    original_processor = AutoImageProcessor.from_pretrained(args.model_path, **processor_kwargs)
    exported_processor = AutoImageProcessor.from_pretrained(args.vit_path, **processor_kwargs)
    original_inputs = original_processor(images=images, return_tensors="pt")
    exported_inputs = exported_processor(images=images, return_tensors="pt")
    for key in ("pixel_values", "image_grid_thw"):
        torch.testing.assert_close(original_inputs[key], exported_inputs[key], atol=0, rtol=0,
                                   msg=f"Export changed preprocessing: {key}")
    inputs = {"pixel_values": original_inputs["pixel_values"].to(device=device, dtype=dtype),
              "image_grid_thw": original_inputs["image_grid_thw"].to(device)}

    print("Loading the unmodified original vision module (including mergers; no LLM)...", flush=True)
    reference = load_original_vision(args.model_path, dtype=dtype, attention=args.attention).to(device).eval()
    captured = []

    def capture_merger_input(module, values):
        captured.append(values[0].detach().cpu().float().clone())

    hook = reference.merger.register_forward_pre_hook(capture_merger_input)
    try:
        with torch.inference_mode():
            reference(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"], return_dict=True)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one original main merger call; captured {len(captured)}")
    expected = captured[0]
    del reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading exported ViT and comparing pre-merger tokens...", flush=True)
    extracted = PreMergerViT.from_export(args.vit_path, dtype=dtype, attention=args.attention).to(device).eval()
    with torch.inference_mode():
        features = extracted(**inputs)
    actual = features.tokens.detach().cpu().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    same_shape = actual.shape == expected.shape
    passed = finite and same_shape and torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    delta = (actual - expected).abs() if finite and same_shape else None
    grid = original_inputs["image_grid_thw"].tolist()
    patch = extracted.vision.config.patch_size
    report = {
        "passed": bool(passed),
        "status": "passed" if passed else "failed",
        "scope": "original_unmodified_vision_main_merger_input_vs_exported_pre_merger_vit",
        "model_path": str(args.model_path.resolve()), "vit_path": str(args.vit_path.resolve()),
        "reference_includes_llm": False,
        "reference_includes_original_mergers": True,
        "preprocessing_exact_match": True,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "device": str(device), "dtype": args.dtype, "attention": args.attention,
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "atol": args.atol, "rtol": args.rtol, "finite": finite,
        "reference_shape": list(expected.shape), "extracted_shape": list(actual.shape),
        "max_abs_error": delta.max().item() if delta is not None else None,
        "mean_abs_error": delta.mean().item() if delta is not None else None,
        "parameter_count": sum(p.numel() for p in extracted.parameters()),
        "images": [
            {"path": str(path.resolve()), "original_wh": list(image.size), "grid_thw": dims,
             "processed_wh": [dims[2] * patch, dims[1] * patch],
             "scale_xy": [dims[2] * patch / image.width, dims[1] * patch / image.height],
             "feature_map_chw": list(feature_map.shape)}
            for path, image, dims, feature_map in zip(args.images, images, grid, features.feature_maps)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("FAIL: extracted features differ. Do not start detection training yet.")
    print("PASS: features match within the recorded tolerances. This is not a detection/mAP evaluation.")


if __name__ == "__main__":
    main()
