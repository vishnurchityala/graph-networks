import torch
from torch import nn

class PCALayer(nn.Module):
    def __init__(self, mean, components):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32)
        components = torch.tensor(components, dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("weight", components)

    def forward(self, x):
        x = x - self.mean
        return torch.matmul(x, self.weight.T)