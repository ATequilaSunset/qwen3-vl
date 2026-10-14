"""Independent Qwen3-VL ViT, retaining the upstream spatial/position handling.

Only safetensors from a local, unquantized dense Qwen3-VL checkpoint are read.
Language-model tensors are never materialized. The upstream forward is reused;
its output merger is replaced by Identity and DeepStack branches are disabled.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import transformers
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


SUPPORTED_TRANSFORMERS = "5.19.0"
EXPORT_FORMAT = "qwen3_vl_pre_merger_v1"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def check_version():
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(
            f"This extraction is verified against transformers=={SUPPORTED_TRANSFORMERS}; "
            f"found {transformers.__version__}. Use vit_probe/requirements.txt in a separate environment."
        )


def _make_vision(config, *, pre_merger, attention):
    check_version()
    if attention not in ("eager", "sdpa"):
        raise ValueError("attention must be eager or sdpa")
    config = deepcopy(config)
    config._attn_implementation = attention
    if pre_merger:
        config.deepstack_visual_indexes = []
    # Skip large random weight initialization. Nonpersistent rotary buffers are
    # still constructed normally on CPU; every parameter is then loaded strictly.
    with no_init_weights(), torch.device("cpu"):
        model = Qwen3VLVisionModel(config)
    if pre_merger:
        model.merger = nn.Identity()
    return model


def _checkpoint_config(path):
    data = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if data.get("model_type") != "qwen3_vl" or "vision_config" not in data:
        raise ValueError("Expected a dense Qwen3-VL checkpoint with vision_config")
    if data.get("quantization_config"):
        raise ValueError("Use the original unquantized checkpoint for this experiment")
    return Qwen3VLVisionConfig.from_dict(data["vision_config"])


def _checkpoint_weights(path, *, pre_merger, dtype):
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        single = path / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"No safetensors checkpoint or index in {path}")
        with safe_open(single, framework="pt", device="cpu") as handle:
            weight_map = dict.fromkeys(handle.keys(), single.name)

    prefixes = [p for p in ("model.visual.", "visual.")
                if p + "patch_embed.proj.weight" in weight_map]
    if len(prefixes) != 1:
        raise ValueError("Cannot identify one unambiguous Qwen3-VL visual weight prefix")
    prefix = prefixes[0]
    by_shard = defaultdict(list)
    skipped = 0
    for full_key, shard in weight_map.items():
        if not full_key.startswith(prefix):
            continue
        key = full_key[len(prefix):]
        if pre_merger and key.startswith(("merger.", "deepstack_merger_list.")):
            skipped += 1
            continue
        by_shard[shard].append((full_key, key))

    state = {}
    for shard, keys in sorted(by_shard.items()):
        shard_path = (path / shard).resolve()
        if not shard_path.is_relative_to(path.resolve()):
            raise ValueError(f"Checkpoint shard is outside model directory: {shard}")
        # Opening a shard reads its header; get_tensor reads ONLY selected tensors,
        # even when the shard also contains language-model weights.
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for full_key, key in keys:
                tensor = handle.get_tensor(full_key)
                if not tensor.is_floating_point():
                    raise ValueError(f"Unsupported non-floating vision tensor: {full_key}")
                state[key] = tensor.to(dtype=dtype)
    return state, {
        "weight_prefix": prefix,
        "loaded_tensor_count": len(state),
        "skipped_merger_tensor_count": skipped,
        "read_shards": sorted(by_shard),
    }


def _load_vision(path, *, pre_merger, dtype, attention):
    path = Path(path).resolve()
    model = _make_vision(_checkpoint_config(path), pre_merger=pre_merger, attention=attention)
    state, report = _checkpoint_weights(path, pre_merger=pre_merger, dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval(), report


def load_original_vision(path, *, dtype=torch.float32, attention="sdpa"):
    """Unmodified upstream vision module, including all mergers, for comparison."""
    return _load_vision(path, pre_merger=False, dtype=dtype, attention=attention)[0]


def tokens_to_feature_maps(tokens, image_grid_thw, merge_size):
    """Undo processor block-major order; return one [C,H,W] map per image.

    A direct tokens.reshape(H,W,C) is WRONG: the processor packs each m*m
    spatial group consecutively even though this backbone does not merge it.
    """
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [num_images, 3]")
    if image_grid_thw.dtype not in (torch.int32, torch.int64):
        raise ValueError("image_grid_thw must contain integer grid sizes")
    grids = image_grid_thw.detach().cpu().tolist()
    if not grids or any(t != 1 or h <= 0 or w <= 0 or h % merge_size or w % merge_size
                        for t, h, w in grids):
        raise ValueError("Only still images (T=1) with positive H,W divisible by merge_size are supported")
    if tokens.ndim != 2 or sum(h * w for _, h, w in grids) != tokens.shape[0]:
        raise ValueError("Token count does not match image_grid_thw")
    maps, offset = [], 0
    for _, h, w in grids:
        block = tokens[offset:offset + h * w]
        block = block.reshape(h // merge_size, w // merge_size, merge_size, merge_size, -1)
        raster = block.permute(0, 2, 1, 3, 4).reshape(h, w, -1)
        maps.append(raster.permute(2, 0, 1).contiguous())
        offset += h * w
    return maps


@dataclass
class ViTFeatures:
    tokens: torch.Tensor  # [sum(H*W), C], original processor block-major order
    feature_maps: list[torch.Tensor]  # each [C,H,W], normal top-to-bottom row order
    image_grid_thw: torch.Tensor


class PreMergerViT(nn.Module):
    """Trainable backbone; freeze parameters explicitly for the frozen experiment."""

    def __init__(self, vision, provenance=None):
        super().__init__()
        self.vision = vision
        self.provenance = provenance or {}

    @classmethod
    def from_qwen_checkpoint(cls, path, *, dtype=torch.float32, attention="sdpa"):
        vision, report = _load_vision(path, pre_merger=True, dtype=dtype, attention=attention)
        return cls(vision, {"source_checkpoint": str(Path(path).resolve()), **report}).eval()

    @classmethod
    def from_export(cls, path, *, dtype=torch.float32, attention="sdpa"):
        path = Path(path)
        metadata = json.loads((path / "extraction.json").read_text(encoding="utf-8"))
        if metadata.get("format") != EXPORT_FORMAT:
            raise ValueError("Unsupported ViT export format")
        config = Qwen3VLVisionConfig.from_dict(
            json.loads((path / "vit_config.json").read_text(encoding="utf-8"))
        )
        vision = _make_vision(config, pre_merger=True, attention=attention)
        state = {k: v.to(dtype=dtype) for k, v in load_file(path / "vit.safetensors").items()}
        vision.load_state_dict(state, strict=True, assign=True)
        return cls(vision, metadata).eval()

    def save_export(self, path, image_processor):
        """Save a standalone backbone plus its original image processor."""
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Output must be absent or empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.vision.state_dict().items()}
        save_file(state, path / "vit.safetensors")
        (path / "vit_config.json").write_text(self.vision.config.to_json_string(), encoding="utf-8")
        image_processor.save_pretrained(path)
        metadata = {
            **self.provenance,
            "format": EXPORT_FORMAT,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "stage": "last_vision_block_before_main_merger",
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "dtype": str(next(self.parameters()).dtype),
            "depth": self.vision.config.depth,
            "hidden_size": self.vision.config.hidden_size,
            "patch_size": self.vision.config.patch_size,
            "spatial_merge_size": self.vision.config.spatial_merge_size,
            "main_merger": False,
            "deepstack_mergers": False,
            "language_model": False,
        }
        (path / "extraction.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return metadata

    def forward(self, pixel_values, image_grid_thw):
        config = self.vision.config
        expected_width = config.in_channels * config.temporal_patch_size * config.patch_size**2
        if pixel_values.ndim != 2 or pixel_values.shape[1] != expected_width:
            raise ValueError(f"pixel_values must be processor-packed [N,{expected_width}], not raw images")
        # Validate the grid before invoking upstream attention/position code.
        tokens_to_feature_maps(pixel_values[:, :1], image_grid_thw, config.spatial_merge_size)
        outputs = self.vision(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        tokens = outputs.last_hidden_state
        return ViTFeatures(tokens, tokens_to_feature_maps(tokens, image_grid_thw,
                                                        config.spatial_merge_size), image_grid_thw)
