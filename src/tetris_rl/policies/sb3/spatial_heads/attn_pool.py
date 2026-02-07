# src/tetris_rl/policies/sb3/spatial_heads/attn_pool.py
from __future__ import annotations

"""
AttentionPoolHead (spatial -> feature vector)

Learned attention pooling over spatial positions (token-free):

  x: (B,H,W,C) -> reshape to (B,N,C) with N=H*W
  logits = score(x) -> (B,N,K)
  w = softmax(logits over N) -> (B,N,K)
  pooled_k = sum_n w[n,k] * x[n] -> (B,K,C)
  pooled = flatten (B, K*C) -> post -> (B,F)
"""

import torch
from torch import nn

from tetris_rl.policies.sb3.spatial_heads.config import AttentionPoolParams
from tetris_rl.policies.sb3.api import Specials, SpatialFeatures
from tetris_rl.policies.sb3.layers.activations import make_activation
from tetris_rl.policies.sb3.spatial_heads.base import BaseSpatialHead


class AttentionPoolHead(BaseSpatialHead):
    def __init__(self, *, in_channels: int, features_dim: int, spec: AttentionPoolParams) -> None:
        super().__init__(features_dim=int(features_dim))

        C = int(in_channels)
        if C <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

        K = int(spec.n_queries)
        if K <= 0:
            raise ValueError("n_queries must be > 0")

        p = float(spec.dropout)
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        self.spec = spec
        self._C = C
        self._K = K

        # logits over positions for each query
        self.score = nn.Linear(C, K, bias=True)

        pooled_in = int(K * C)

        # post: always outputs features_dim (either via MLP or direct projection)
        if int(spec.mlp_hidden) > 0:
            Hh = int(spec.mlp_hidden)
            if Hh <= 0:
                raise ValueError("mlp_hidden must be > 0 when enabled")
            self.post = nn.Sequential(
                nn.Linear(pooled_in, Hh),
                make_activation(spec.activation),
                nn.Dropout(p) if p > 0.0 else nn.Identity(),
                nn.Linear(Hh, self.features_dim),
            )
        else:
            self.post = nn.Identity() if pooled_in == self.features_dim else nn.Linear(pooled_in, self.features_dim)

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> torch.Tensor:
        _ = specials
        x = self._check_spatial(spatial)  # (B,H,W,C)
        B, H, W, C = x.shape
        if int(C) != int(self._C):
            raise ValueError(f"in_channels mismatch: head expects C={self._C}, got C={int(C)}")

        seq = x.reshape(int(B), int(H) * int(W), int(C))  # (B,N,C)
        logits = self.score(seq)  # (B,N,K)
        w = torch.softmax(logits, dim=1)  # over positions N

        pooled = torch.einsum("bnk,bnc->bkc", w, seq)  # (B,K,C)
        pooled = pooled.reshape(int(B), int(self._K) * int(C))  # (B, K*C)

        out = self.post(pooled)
        return self._check_out(out)


__all__ = ["AttentionPoolHead"]


