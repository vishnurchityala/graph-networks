import torch
from torch import nn

class ClassificationLayer(nn.Module):
    def __init__(self, input_dim: int = 50+50+64, hidden_dim: int = 64, out_dim: int = 4, dropout_prob: float = 0.75):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.classifier(x)
