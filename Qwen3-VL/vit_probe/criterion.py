"""Bipartite matching and CE + L1 + generalized IoU losses, including auxiliaries."""

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def generalized_iou(a, b):
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    enclosure = (torch.maximum(a[:, None, 2:], b[None, :, 2:]) -
                 torch.minimum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    return intersection / union.clamp(min=1e-7) - (enclosure - union) / enclosure.clamp(min=1e-7)


@torch.no_grad()
def overlaps_ignore(boxes, ignored, threshold=.5):
    """Ignore unmatched queries whose visible box is >=50% in an ignored region."""
    if ignored is None or not ignored.numel():
        return torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    a = cxcywh_to_xyxy(boxes).clamp(0, 1)
    b = cxcywh_to_xyxy(ignored).clamp(0, 1)
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    area = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    return (intersection / area[:, None].clamp(min=1e-9)).max(1).values >= threshold


class SetCriterion(nn.Module):
    def __init__(self, num_classes, class_weight=1., box_weight=5., giou_weight=2., eos_weight=.1):
        super().__init__()
        self.num_classes = num_classes
        self.weights = {"ce": class_weight, "l1": box_weight, "giou": giou_weight}
        weight = torch.ones(num_classes + 1)
        weight[-1] = eos_weight
        self.register_buffer("class_weights", weight)

    @torch.no_grad()
    def match(self, logits, boxes, targets):
        matches = []
        for pred_logits, pred_boxes, target in zip(logits, boxes, targets):
            if len(target["labels"]) > len(pred_boxes):
                raise ValueError("More ground-truth objects than queries; increase num_queries. "
                                 "Targets must never be silently truncated.")
            cost = -self.weights["ce"] * pred_logits.softmax(-1)[:, target["labels"]]
            cost = cost + self.weights["l1"] * torch.cdist(pred_boxes, target["boxes"], p=1)
            cost = cost - self.weights["giou"] * generalized_iou(
                cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(target["boxes"]))
            if not torch.isfinite(cost).all():
                raise FloatingPointError("Non-finite Hungarian matching cost")
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            matches.append((torch.as_tensor(rows, dtype=torch.long, device=boxes.device),
                            torch.as_tensor(cols, dtype=torch.long, device=boxes.device)))
        return matches

    def one_layer(self, output, targets):
        # Matching, softmax, distances and GIoU are computed in FP32 even in AMP.
        logits, boxes = output["pred_logits"].float(), output["pred_boxes"].float()
        matches = self.match(logits, boxes, targets)
        labels = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        pred, truth = [], []
        for i, (rows, cols) in enumerate(matches):
            labels[i, overlaps_ignore(boxes[i], targets[i].get("ignore_boxes"))] = -100
            # Valid GT matches always take precedence over ignored background.
            labels[i, rows] = targets[i]["labels"][cols]
            pred.append(boxes[i, rows])
            truth.append(targets[i]["boxes"][cols])
        pred, truth = torch.cat(pred), torch.cat(truth)
        count = max(sum(len(t["labels"]) for t in targets), 1)
        return {
            "ce": (F.cross_entropy(logits.transpose(1, 2), labels, self.class_weights, ignore_index=-100)
                   if (labels != -100).any() else logits.sum() * 0),
            "l1": F.l1_loss(pred, truth, reduction="sum") / count,
            "giou": (1 - generalized_iou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(truth)).diag()).sum() / count,
        }

    def forward(self, output, targets):
        with torch.autocast(device_type=output["pred_boxes"].device.type, enabled=False):
            metrics = self.one_layer(output, targets)
            total = sum(self.weights[k] * value for k, value in metrics.items())
            for i, auxiliary in enumerate(output.get("aux_outputs", [])):
                losses = self.one_layer(auxiliary, targets)
                total = total + sum(self.weights[k] * value for k, value in losses.items())
                metrics.update({f"{key}_aux{i}": value for key, value in losses.items()})
        return total, metrics
