import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_feature = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device = device, dtype = dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        std = (2/(self.in_feature + self.out_features))**0.5
        mean = 0
        nn.init.trunc_normal_(self.weight, mean = mean, std = std, a = - 3*std, b = 3*std)
        
    def forward(self, x):
        return einsum(x, self.weight, "... i, o i -> ... o")
