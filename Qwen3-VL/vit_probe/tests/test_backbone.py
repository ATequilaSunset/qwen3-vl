"""Offline checks on a tiny real upstream Qwen3-VL model, not mocked layers."""

import json
import sys

import numpy as np
from PIL import Image
import pytest
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoImageProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from vit_probe.backbone import PreMergerViT, load_original_vision, tokens_to_feature_maps


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(23)
    torch.set_num_threads(1)
    vision_config = Qwen3VLVisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
        patch_size=4, temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0],
    )
    config = Qwen3VLConfig(
        vision_config=vision_config.to_dict(),
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=1024),
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
    )
    config._attn_implementation = "eager"
    original = Qwen3VLForConditionalGeneration(config).eval()
    model_path = tmp_path / "qwen"
    original.save_pretrained(model_path)
    processor = Qwen2VLImageProcessor(patch_size=4, temporal_patch_size=2, merge_size=2,
                                     min_pixels=64, max_pixels=1024)
    processor.save_pretrained(model_path)
    rng = np.random.default_rng(11)
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
              for h, w in ((16, 24), (32, 16))]
    inputs = processor(images=images, return_tensors="pt")
    return model_path, original, processor, images, inputs


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_original_hook_and_export_roundtrip(checkpoint, tmp_path, attention):
    path, full_model, processor, images, inputs = checkpoint
    expected = []
    hook = full_model.model.visual.merger.register_forward_pre_hook(
        lambda module, args: expected.append(args[0].detach().clone())
    )
    with torch.no_grad():
        full_model.model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    hook.remove()
    backbone = PreMergerViT.from_qwen_checkpoint(path, attention=attention)
    with torch.no_grad():
        actual = backbone(**inputs)
    torch.testing.assert_close(actual.tokens, expected[0], atol=1e-6, rtol=1e-5)
    assert [list(t.shape) for t in actual.feature_maps] == [[32, 4, 6], [32, 8, 4]]
    assert all("merger" not in key and "language_model" not in key for key in backbone.state_dict())

    exported = tmp_path / "vit"
    backbone.save_export(exported, processor)
    reload_processor = AutoImageProcessor.from_pretrained(exported, local_files_only=True)
    reload_inputs = reload_processor(images=images, return_tensors="pt")
    torch.testing.assert_close(reload_inputs.pixel_values, inputs.pixel_values, atol=0, rtol=0)
    reloaded = PreMergerViT.from_export(exported, attention=attention)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**reload_inputs).tokens, actual.tokens, atol=0, rtol=0)


def test_shards_do_not_require_language_model_file(checkpoint):
    path, original, _, _, inputs = checkpoint
    state = load_file(path / "model.safetensors")
    visual = [(k, v) for k, v in state.items() if k.startswith("model.visual.")]
    weight_map = {k: "missing-language-shard.safetensors" for k in state if not k.startswith("model.visual.")}
    for i in range(2):
        shard = f"visual-{i}.safetensors"
        selected = dict(visual[i::2])
        save_file(selected, path / shard)
        weight_map.update(dict.fromkeys(selected, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "model.safetensors").unlink()
    extracted = PreMergerViT.from_qwen_checkpoint(path, attention="eager")
    reference = load_original_vision(path, attention="eager")
    with torch.no_grad():
        expected = original.model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw).last_hidden_state
        torch.testing.assert_close(extracted(**inputs).tokens, expected, atol=0, rtol=0)
        torch.testing.assert_close(reference(inputs.pixel_values, grid_thw=inputs.image_grid_thw).last_hidden_state,
                                   expected, atol=0, rtol=0)
    assert extracted.provenance["read_shards"] == ["visual-0.safetensors", "visual-1.safetensors"]


def test_spatial_mapping_and_batch_boundaries():
    # Explicit upstream 2x2 block ordering of a 4x6 raster.
    block_order = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11,
                   12, 13, 18, 19, 14, 15, 20, 21, 16, 17, 22, 23]
    tokens = torch.tensor(block_order + [100, 101, 102, 103], dtype=torch.float32)[:, None]
    maps = tokens_to_feature_maps(tokens, torch.tensor([[1, 4, 6], [1, 2, 2]]), 2)
    torch.testing.assert_close(maps[0], torch.arange(24, dtype=torch.float32).reshape(1, 4, 6))
    torch.testing.assert_close(maps[1], torch.arange(100, 104, dtype=torch.float32).reshape(1, 2, 2))
    with pytest.raises(ValueError, match="still images"):
        tokens_to_feature_maps(tokens, torch.tensor([[2, 2, 2]]), 2)
    with pytest.raises(ValueError, match="Token count"):
        tokens_to_feature_maps(tokens, torch.tensor([[1, 2, 2]]), 2)


def test_gradients_for_finetuned_and_frozen_backbone(checkpoint):
    path, _, _, _, inputs = checkpoint
    backbone = PreMergerViT.from_qwen_checkpoint(path, attention="eager").train()
    features = backbone(**inputs)
    features.feature_maps[0].square().mean().backward()
    grad = backbone.vision.patch_embed.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    backbone.zero_grad(set_to_none=True)
    backbone.requires_grad_(False)
    head = torch.nn.Conv2d(32, 8, 1)
    head(backbone(**inputs).feature_maps[0].unsqueeze(0)).square().mean().backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_missing_parameter_is_not_silently_initialized(checkpoint):
    path, _, _, _, _ = checkpoint
    state = load_file(path / "model.safetensors")
    del state["model.visual.blocks.0.norm1.weight"]
    save_file(state, path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        PreMergerViT.from_qwen_checkpoint(path)


def test_cli_pass_and_mismatch_failure(checkpoint, tmp_path, monkeypatch):
    from vit_probe import extract, verify

    path, _, _, images, _ = checkpoint
    exported = tmp_path / "exported"
    monkeypatch.setattr(sys, "argv", ["extract", "--model-path", str(path), "--output-dir", str(exported)])
    extract.main()
    image_path = tmp_path / "image.png"
    images[0].save(image_path)
    output = tmp_path / "comparison.json"
    argv = ["verify", "--model-path", str(path), "--vit-path", str(exported),
            "--images", str(image_path), "--output", str(output),
            "--min-pixels", "64", "--max-pixels", "1024"]
    monkeypatch.setattr(sys, "argv", argv)
    verify.main()
    report = json.loads(output.read_text())
    assert report["passed"] and report["max_abs_error"] == 0
    assert report["images"][0]["processed_wh"] == [24, 16]

    state = load_file(exported / "vit.safetensors")
    state["patch_embed.proj.bias"][0] += 1
    save_file(state, exported / "vit.safetensors")
    with pytest.raises(SystemExit, match="FAIL"):
        verify.main()
    assert json.loads(output.read_text())["passed"] is False
