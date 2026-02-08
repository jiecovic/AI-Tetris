# src/planning_rl/td/model.py
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LinearValueModel(nn.Module):
    def __init__(self, *, num_features: int) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be >= 1")
        self.weights = nn.Parameter(torch.zeros((int(num_features),), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or int(x.shape[1]) != int(self.weights.shape[0]):
            raise ValueError("input must be (B,F) with F == num_features")
        return (x * self.weights).sum(dim=1)

    def get_weights(self) -> list[float]:
        return [float(w) for w in self.weights.detach().cpu().tolist()]

    def set_weights(self, weights: Sequence[float]) -> None:
        if len(weights) != int(self.weights.shape[0]):
            raise ValueError("weights length mismatch")
        with torch.no_grad():
            self.weights.copy_(
                torch.tensor(list(weights), dtype=self.weights.dtype, device=self.weights.device)
            )


__all__ = ["LinearValueModel"]
