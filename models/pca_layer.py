import torch
from torch import nn

class PCALayer(nn.Module):
    def __init__(self, mean, components, output_dim: int | None = None):
        """
        PCA projection layer.

        - `mean`: 1D array-like of shape (in_dim,)
        - `components`: 2D array-like of shape (n_components, in_dim)
        - `output_dim`: if provided, use only the first `output_dim` components
                        (so the layer outputs `output_dim`-dimensional vectors).
        """
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32)
        components = torch.tensor(components, dtype=torch.float32)

        if output_dim is not None:
            if output_dim > components.shape[0]:
                raise ValueError(
                    f"Requested output_dim={output_dim}, "
                    f"but only {components.shape[0]} PCA components are available."
                )
            components = components[:output_dim]

        self.register_buffer("mean", mean)
        self.register_buffer("weight", components)
        self.output_dim = self.weight.shape[0]

    def forward(self, x):
        x = x - self.mean
        return torch.matmul(x, self.weight.T)