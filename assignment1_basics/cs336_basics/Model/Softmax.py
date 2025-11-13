import torch

def Softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of tensor x along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax. Default is -1.

    Returns:
        torch.Tensor: Tensor after applying softmax.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x