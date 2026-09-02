import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.025) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class BalancedLossMLP(nn.Module):
    """MLP used for ASD-vs-TD classification from selected FC edges."""

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.72,
        noise_std: float = 0.025,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(noise_std)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.noise(x))

