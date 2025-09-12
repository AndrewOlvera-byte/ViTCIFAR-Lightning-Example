from __future__ import annotations

from typing import Optional

import torch.nn as nn

from src.registry import register_loss


@register_loss("cross_entropy")
def build_cross_entropy(label_smoothing: float = 0.0, reduction: str = "mean") -> nn.Module:
    # PyTorch supports label_smoothing in CrossEntropyLoss
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)


