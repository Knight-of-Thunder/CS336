import torch
import torch.nn as nn
from torch import Tensor
from .Embedding import Embedding
from .RMSNorm import RMSNorm
from .Linear import Linear
from .TransformerBlock import TransformerBlock
from einops import repeat

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=True,
                theta=rope_theta,
                max_seq_len=context_length
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
    def load_state_dict(self, state_dict):
        """Load a state dictionary with a different naming convention."""
        new_state_dict = {}
        
        # Map the main components
        if "token_embeddings.weight" in state_dict:
            new_state_dict["token_embedding.weight"] = state_dict["token_embeddings.weight"]
        if "lm_head.weight" in state_dict:
            new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
        if "ln_final.weight" in state_dict:
            new_state_dict["ln_final.scale"] = state_dict["ln_final.weight"]
            
        # Map each transformer block
        for i in range(len(self.blocks)):
            prefix = f"layers.{i}."
            new_prefix = f"blocks.{i}."
            
            # Map attention weights
            new_state_dict[new_prefix + "mhsa.W_Q.weight"] = state_dict[prefix + "attn.q_proj.weight"]
            new_state_dict[new_prefix + "mhsa.W_K.weight"] = state_dict[prefix + "attn.k_proj.weight"]
            new_state_dict[new_prefix + "mhsa.W_V.weight"] = state_dict[prefix + "attn.v_proj.weight"]
            new_state_dict[new_prefix + "mhsa.W_O.weight"] = state_dict[prefix + "attn.output_proj.weight"]
            
            # Map FFN weights
            new_state_dict[new_prefix + "ffn.linear1.weight"] = state_dict[prefix + "ffn.w1.weight"]
            new_state_dict[new_prefix + "ffn.linear2.weight"] = state_dict[prefix + "ffn.w2.weight"]
            new_state_dict[new_prefix + "ffn.linear3.weight"] = state_dict[prefix + "ffn.w3.weight"]
            
            # Map norm weights
            new_state_dict[new_prefix + "norm1.scale"] = state_dict[prefix + "ln1.weight"]
            new_state_dict[new_prefix + "norm2.scale"] = state_dict[prefix + "ln2.weight"]
            
        # Call the parent class's load_state_dict with our remapped dictionary
        super().load_state_dict(new_state_dict)

    def forward(self, input_indices: Tensor) -> Tensor:
        b, s = input_indices.shape
        pos_1d = torch.arange(s) 
        pos = repeat(pos_1d, 's -> b s', b=b)
        x = self.token_embedding(input_indices)
        for block in self.blocks:
            x = block(x, pos)
        x = self.ln_final(x)
        return self.lm_head(x)
