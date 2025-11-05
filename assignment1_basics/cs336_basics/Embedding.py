import torch
import torch.nn as nn
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1
        mean = 0
        nn.init.trunc_normal_(self.weight, mean = mean, std = std, a = - 3*std, b = 3*std)

    def forward(self, x):
        return self.weight[x]