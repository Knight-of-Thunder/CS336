import torch
import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Int
from torch import Tensor
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Float[Tensor, " batch_size context_length"], Float[Tensor, " batch_size context_length"]]:
    """
    Given a dataset, batch size, and context length, return a batch of data.
    """
    N = len(dataset)
    starts = np.random.randint(0, N - context_length, batch_size)

    inputs = torch.empty((batch_size, context_length), dtype=torch.long)
    targets = torch.empty((batch_size, context_length), dtype=torch.long)

    for i , start in enumerate(starts):
        inputs[i] = torch.from_numpy(dataset[start:start+context_length])
        targets[i] = torch.from_numpy(dataset[start+1:start+context_length+1])

    return inputs.to(device), targets.to(device)