import torch
import torch.nn as nn
from einops import rearrange
from .scaled_dot_product_attention import scaled_dot_product_attention
from .Linear import Linear
from jaxtyping import Int

class multihead_self_attention(nn.Module):
    def __init__(
        self,     
        d_model: int,
        num_heads: int,
        rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,

    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = self.d_k = d_model // num_heads

        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

        if rope:
            from .RoPE import RotaryPositionalEmbedding as RoPE
            assert max_seq_len is not None
            assert theta is not None
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ):
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        if hasattr(self, 'rope'):
            assert token_positions is not None
            Q, K = self.rope.forward(Q, token_positions), self.rope.forward(K, token_positions)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        output = self.W_O(attn_output)
        return output
