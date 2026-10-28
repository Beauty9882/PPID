import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityPerturbator(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int | None = None, dropout_p: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.residual = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.dropout(self.residual(z))
        return F.normalize(z + delta, dim=-1)


PerturbationGenerator = IdentityPerturbator
