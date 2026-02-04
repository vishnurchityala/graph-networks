import torch
from torch import nn

class LDALayer(nn.Module):
    def __init__(self, mean, coef):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("weight", torch.tensor(coef, dtype=torch.float32))  # coef_: (C-1, D)

    def forward(self, x):
        x = x - self.mean
        x = torch.matmul(x, self.weight.T)
        return x