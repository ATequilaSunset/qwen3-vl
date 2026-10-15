"""Query-based detection on the LAST pre-merger ViT feature map.

This is a single-scale DETR-style diagnostic, not an implementation of the
full published DETR backbone/encoder or an official AI-TOD benchmark model.
"""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def position_encoding(h, w, dim, device):
    """2-D sine/cosine coordinates, separately normalized for each image."""
    if dim % 4:
        raise ValueError("hidden_dim must be divisible by 4")
    frequency = 10000 ** (torch.arange(dim // 4, device=device).float() / (dim // 4))
    y, x = torch.meshgrid((torch.arange(h, device=device).float() + .5) / h,
                          (torch.arange(w, device=device).float() + .5) / w, indexing="ij")
    x = x.flatten()[:, None] * (2 * math.pi) / frequency
    y = y.flatten()[:, None] * (2 * math.pi) / frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)


class QueryLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, query_position, memory, position, mask):
        q = k = target + query_position
        update = self.self_attention(q, k, target, need_weights=False)[0]
        target = self.norms[0](target + self.dropout(update))
        update = self.cross_attention(target + query_position, memory + position, memory,
                                      key_padding_mask=mask, need_weights=False)[0]
        target = self.norms[1](target + self.dropout(update))
        return self.norms[2](target + self.dropout(self.ffn(target)))


class DetectionHead(nn.Module):
    def __init__(self, input_dim, num_classes, num_queries=300, hidden_dim=256,
                 decoder_layers=6, heads=8, ffn_dim=1024, dropout=.1):
        super().__init__()
        if min(input_dim, num_classes, num_queries, decoder_layers, heads) < 1:
            raise ValueError("Detection dimensions must be positive")
        if hidden_dim % 4 or hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by 4 and heads")
        self.num_queries, self.hidden_dim = num_queries, hidden_dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.queries = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList([QueryLayer(hidden_dim, heads, ffn_dim, dropout)
                                     for _ in range(decoder_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)  # final class = no object
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 4))

    def forward(self, feature_maps):
        # feature_maps ALREADY undo Qwen's block-major token ordering.
        sequences = [self.projection(f.flatten(1).transpose(0, 1)) for f in feature_maps]
        memory = pad_sequence(sequences, batch_first=True)
        positions = [position_encoding(f.shape[-2], f.shape[-1], self.hidden_dim, f.device)
                     for f in feature_maps]
        position = pad_sequence(positions, batch_first=True).to(memory.dtype)
        lengths = torch.tensor([len(s) for s in sequences], device=memory.device)
        mask = torch.arange(memory.shape[1], device=memory.device)[None] >= lengths[:, None]
        query_position = self.queries.weight[None].expand(len(feature_maps), -1, -1)
        target = torch.zeros_like(query_position)
        outputs = []
        for layer in self.layers:
            target = layer(target, query_position, memory, position, mask)
            outputs.append({"pred_logits": self.classifier(target),
                            "pred_boxes": self.box_head(target).sigmoid()})
        return {**outputs[-1], "aux_outputs": outputs[:-1]}


class ViTDetector(nn.Module):
    def __init__(self, backbone, head, frozen):
        super().__init__()
        self.backbone, self.head, self.frozen = backbone, head, frozen
        self.backbone.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values, image_grid_thw):
        # no_grad, NOT inference_mode: the downstream trainable head needs to
        # save these features for its own backward pass.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features = self.backbone(pixel_values, image_grid_thw).feature_maps
        return self.head(features)


def build_detector(vit_path, head_config, frozen, gradient_checkpointing=False):
    from .backbone import PreMergerViT
    backbone = PreMergerViT.from_export(vit_path)
    head = DetectionHead(input_dim=backbone.vision.config.hidden_size, **head_config)
    model = ViTDetector(backbone, head, frozen)
    if gradient_checkpointing and not frozen:
        backbone.vision.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model
