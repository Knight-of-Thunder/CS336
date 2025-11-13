import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = x.norm(2,dim = -1, keepdim = True)
        rms_x2 = norm_x.pow(2)/x.shape[-1] + self.eps
        rms_x = rms_x2.sqrt()
        x_normed = x / rms_x
        x_scaled = x_normed * self.scale
        return x_scaled.to(in_dtype)