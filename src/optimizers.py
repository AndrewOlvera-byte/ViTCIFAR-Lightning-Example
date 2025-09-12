from __future__ import annotations

from typing import Any, Iterable, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

from src.registry import register_optimizer, register_scheduler


@register_optimizer("adamw")
def build_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    fused: bool = False,
) -> Optimizer:
    extra: dict = {}
    if fused and torch.cuda.is_available():
        try:
            extra["fused"] = True
        except TypeError:
            # Older torch without fused AdamW support
            pass
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, **extra)


@register_scheduler("cosine")
def build_cosine(optimizer: Optimizer, t_max: int, eta_min: float) -> _LRScheduler:
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


