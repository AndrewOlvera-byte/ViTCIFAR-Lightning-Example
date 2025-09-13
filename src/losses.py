from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch

from src.registry import register_loss


@register_loss("cross_entropy")
def build_cross_entropy(label_smoothing: float = 0.0, reduction: str = "mean") -> nn.Module:
    # PyTorch supports label_smoothing in CrossEntropyLoss
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy with soft targets (e.g., Mixup/CutMix).

    Expects logits of shape (B, C) and targets of shape (B, C) with rows summing to 1.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("soft_cross_entropy")
def build_soft_cross_entropy(reduction: str = "mean") -> nn.Module:
    return SoftTargetCrossEntropy(reduction=reduction)

