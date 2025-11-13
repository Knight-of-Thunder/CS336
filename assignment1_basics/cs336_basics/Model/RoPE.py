import torch
import torch.nn as nn
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be an even number")
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1 /  self.theta ** (torch.arange(0, self.d_k, 2)  / self.d_k)

        t = torch.arange(max_seq_len)

        angles = einsum(t, freqs, "i, j -> i j")

        cos_cache = angles.cos()
        sin_cache = angles.sin()

        cos_cache_interleave = torch.repeat_interleave(cos_cache, 2, dim = -1)
        sin_cache_interleave = torch.repeat_interleave(sin_cache, 2, dim = -1)
        
        self.register_buffer("cos_cached", cos_cache_interleave, persistent=False)
        self.register_buffer("sin_cached", sin_cache_interleave, persistent=False)

    def _rotate_adjacent_pairs(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x2_neg = -x2

        return torch.stack([x2_neg, x1], dim = -1).flatten(-2)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        cos = self.cos_cached[token_positions.to(torch.long)]
        sin = self.sin_cached[token_positions.to(torch.long)]
        x_paired = self._rotate_adjacent_pairs(x)

        x_rotated = x * cos + x_paired * sin
        return x_rotated




