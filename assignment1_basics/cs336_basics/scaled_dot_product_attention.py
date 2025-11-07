from einops import einsum
from .Softmax import Softmax

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    QK = einsum(query, key, "... q d_k, ... k d_k -> ... q k")
    d_k = query.shape[-1]
    scaled_QK = QK / d_k**0.5
    if mask is not None:
        scaled_QK = scaled_QK.masked_fill(~mask, float('-inf'))
    attention_weights = Softmax(scaled_QK, dim=-1)
    output = einsum(attention_weights, value, "... q k, ... k d_v -> ... q d_v")
    return output