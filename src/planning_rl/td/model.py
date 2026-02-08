# src/planning_rl/td/model.py
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LinearValueModel(nn.Module):
    def __init__(
        self,
        *,
        num_features: int,
        weight_norm: str = "none",
        weight_scale: float = 1.0,
        learn_scale: bool = True,
        weight_norm_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be >= 1")
        norm = str(weight_norm).strip().lower()
        if norm in {"none", "off", "false", "0"}:
            norm = "none"
        elif norm in {"l2", "unit", "unit_l2", "unit_sphere", "sphere"}:
            norm = "l2"
        else:
            raise ValueError(f"unknown weight_norm: {weight_norm!r}")

        self.weight_norm = norm
        self.learn_scale = bool(learn_scale)
        self.weight_norm_eps = float(weight_norm_eps)

        self.weights = nn.Parameter(torch.zeros((int(num_features),), dtype=torch.float32))
        if self.weight_norm == "l2":
            scale = torch.tensor(float(weight_scale), dtype=torch.float32)
            if self.learn_scale:
                self.weight_scale = nn.Parameter(scale)
            else:
                self.register_buffer("weight_scale", scale)

    def _effective_weights(self) -> torch.Tensor:
        if self.weight_norm != "l2":
            return self.weights
        norm = torch.norm(self.weights, p=2)
        w_unit = self.weights / (norm + float(self.weight_norm_eps))
        return w_unit * self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or int(x.shape[1]) != int(self.weights.shape[0]):
            raise ValueError("input must be (B,F) with F == num_features")
        w = self._effective_weights()
        return (x * w).sum(dim=1)

    def get_weights(self) -> list[float]:
        w = self._effective_weights()
        return [float(v) for v in w.detach().cpu().tolist()]

    def set_weights(self, weights: Sequence[float]) -> None:
        if len(weights) != int(self.weights.shape[0]):
            raise ValueError("weights length mismatch")
        w = torch.tensor(list(weights), dtype=self.weights.dtype, device=self.weights.device)
        with torch.no_grad():
            if self.weight_norm == "l2":
                norm = torch.norm(w, p=2)
                self.weights.copy_(w / (norm + float(self.weight_norm_eps)))
                scale_val = float(norm.detach().cpu().item()) if norm.numel() == 1 else float(norm)
                if hasattr(self, "weight_scale"):
                    self.weight_scale.copy_(
                        torch.tensor(scale_val, dtype=self.weights.dtype, device=self.weights.device)
                    )
            else:
                self.weights.copy_(w)


__all__ = ["LinearValueModel"]
