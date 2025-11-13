from typing import Iterable
import torch
import math
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm_squared = 0
    for p in parameters:
        if p.grad is not None:
            total_norm_squared += p.grad.data.pow(2).sum()
    total_norm = math.sqrt(total_norm_squared)
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= clip_coef