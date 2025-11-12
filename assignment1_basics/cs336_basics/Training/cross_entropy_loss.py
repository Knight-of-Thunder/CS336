
import torch
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"] 
) -> Float[Tensor, ""]:
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    targets = rearrange(targets, " ... -> ... 1")
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)

    loss = -target_log_probs.mean()
    return loss