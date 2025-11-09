import torch
import torch.nn as nn

from .RMSNorm import RMSNorm
from .multihead_self_attention import multihead_self_attention
from .SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: bool = False, theta: float = 0, max_seq_len: int = 2048):
        super().__init__()
        self.mhsa = multihead_self_attention(d_model=d_model, num_heads=num_heads, rope=rope, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        attn_output = self.mhsa(self.norm1(x), token_positions=token_positions)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x