import torch
from Model.Softmax import Softmax
def decode(model, prompt, max_new_tokens = 50, temperature = 0.9, top_p = 0.9, eos_token_id = None):
    """
    Decode the prompt using the model.
    """
    out_put = prompt.clone()
    for i in range(max_new_tokens):
        logits = model(out_put)[..., -1, :] / temperature
        logits = Softmax(logits)
        logits = top_p_filter(logits, top_p)
        next_token = torch.multinomial(logits, num_samples = 1)
        out_put = torch.cat([out_put, next_token], dim = -1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return out_put

def top_p_filter(probs, top_p = 0.9): 
    """
    Filter the logits using top-p filtering.
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cumulative_probs, top_p)
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask[sorted_indices[:cutoff+1]] = False
    probs = probs.clone()
    probs[mask] = 0.0
    probs /= probs.sum()
    return probs
