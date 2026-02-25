import torch
from torch import nn

class ClassificationLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 50 + 50 + 64,
        hidden_dims=(256, 128, 64),
        out_dim: int = 4,
        dropout_prob: float = 0.3
    ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.out = nn.Linear(hidden_dims[2], out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)
