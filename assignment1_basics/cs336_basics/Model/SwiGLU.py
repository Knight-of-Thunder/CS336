import torch
import torch.nn as nn
from torch import sigmoid
from .Linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.linear3 = Linear(d_model, d_ff)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_result =  self.linear1.forward(x) * sigmoid(self.linear1.forward(x)) * self.linear3.forward(x)
        return self.linear2.forward(half_result)