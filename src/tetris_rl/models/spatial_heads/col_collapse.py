# src/tetris_rl/models/spatial_heads/col_collapse.py
from __future__ import annotations

"""
ColumnCollapseHead (spatial -> feature vector)

Faithful reconstruction of the column-wise convolutional architecture
described in a Tetris reinforcement learning project report
(assuming the 2D conv stem conv3-32, conv3-32, conv3-64 is implemented upstream).

Report architecture (after stem):
  - a layer that collapses each column into a single "pixel" with 64 channels
  - conv3-128, conv1-128, conv3-128 applied over columns
  - FC-128, FC-512, then a task-specific head (FC-13 or FC-1)

In SB3, this module stops at the feature extractor output, so we target:
  output features_dim = 512
and rely on SB3's policy/value heads for the final logits/value.

Pipeline (this module)
----------------------
Input:
  SpatialFeatures.x: (B, H, W, C)  channel-last

1) Column collapse (reduce over H):
  (B,H,W,C) -> (B,W,C)

   The report refers to this step only as a "layer" and does not specify
   its exact form. We therefore make this stage configurable:

     - "linear" : learned weighted sum over height (default; closest to
                  an unspecified learned layer)
     - "avg"    : mean pooling over H
     - "max"    : max pooling over H

2) 1D convolutional stack over width W:
  (B,W,C) -> permute -> (B,C,W)
  conv k=3 -> 128
  conv k=1 -> 128
  conv k=3 -> 128
  (with BatchNorm + ReLU + Dropout(p=0.25) after each conv)

3) Pool over columns (W axis):
  avg / max / avgmax  -> (B, 128) or (B, 256)

4) (Optional) include piece one-hots (from Specials):
  concat active_onehot and/or next_onehot to pooled vector (before FC stack)

5) Fully connected stack:
  FC-128 -> ReLU -> Dropout
  FC-512 -> ReLU -> Dropout
  (optional output projection if features_dim != 512)

Notes
-----
- The report states a dropout *retention probability* of 0.75,
  which corresponds to dropout p = 0.25.
- The report does not specify the exact form of the column-collapse layer.
  The default implementation assumes a learned linear reduction over height; alternative fixed reductions (avg/max) are supported for ablations.
- `Specials` is accepted for interface uniformity and can optionally provide
  active/next piece IDs for one-hot augmentation.
"""

from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F

from tetris_rl.models.spatial_heads.config import ColumnCollapseParams
from tetris_rl.models.api import Specials, SpatialFeatures
from tetris_rl.models.layers.activations import make_activation
from tetris_rl.models.spatial_heads.base import BaseSpatialHead

CollapseKind = Literal["avg", "max", "linear"]
Pool1D = Literal["avg", "max", "avgmax"]


def _normalize_collapse(params: ColumnCollapseParams) -> CollapseKind:
    """
    Normalize collapse mode.
    """
    raw = getattr(params, "collapse", None)
    s = str(raw).strip().lower()
    if s in {"avg", "mean"}:
        return "avg"
    if s in {"max"}:
        return "max"
    if s in {"linear"}:
        return "linear"
    # allow some accidental values from older experiments
    if s in {"meanmax", "avgmax"}:
        # meanmax is NOT a collapse; treat it as invalid here so configs don't silently change semantics
        raise ValueError("collapse cannot be meanmax/avgmax; use pool='avgmax' for column pooling instead")
    raise ValueError(f"collapse must be avg|max|linear (or mean as alias), got {raw!r}")


def _normalize_pool(params: ColumnCollapseParams) -> Pool1D:
    s = str(getattr(params, "pool", "avg")).strip().lower()
    if s not in {"avg", "max", "avgmax"}:
        raise ValueError(f"pool must be avg|max|avgmax, got {s!r}")
    return s  # type: ignore[return-value]


class _LinearHeightCollapse(nn.Module):
    """
    Learned column collapse over height H.

    Input:  x (B,H,W,C)
    Output: y (B,W,C)

    Implements a learned weighted sum over H for each channel (or shared over channels):
      y[b,w,c] = sum_h weight[c,h] * x[b,h,w,c] + bias[c]   (per-channel)
    """

    def __init__(self, *, H: int, C: int, per_channel: bool) -> None:
        super().__init__()
        self.H = int(H)
        self.C = int(C)
        self.per_channel = bool(per_channel)

        if self.H <= 0 or self.C <= 0:
            raise ValueError("H and C must be > 0")

        if self.per_channel:
            self.weight = nn.Parameter(torch.zeros((self.C, self.H), dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros((self.C,), dtype=torch.float32))
        else:
            self.weight = nn.Parameter(torch.zeros((self.H,), dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

        # start as near-average (stable init)
        with torch.no_grad():
            if self.per_channel:
                self.weight.fill_(1.0 / float(self.H))
            else:
                self.weight.fill_(1.0 / float(self.H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be (B,H,W,C), got {tuple(x.shape)}")
        _B, H, _W, C = x.shape
        if int(H) != int(self.H) or int(C) != int(self.C):
            raise ValueError(f"shape changed: expected H={self.H},C={self.C}, got H={int(H)},C={int(C)}")

        # x: (B,H,W,C) -> (B,W,C,H)
        x_bwch = x.permute(0, 2, 3, 1).contiguous()

        if self.per_channel:
            # weight: (C,H) -> broadcast to (1,1,C,H)
            w = self.weight.view(1, 1, self.C, self.H)
            y = (x_bwch * w).sum(dim=-1) + self.bias.view(1, 1, self.C)  # (B,W,C)
        else:
            # weight: (H,) -> broadcast to (1,1,1,H)
            w = self.weight.view(1, 1, 1, self.H)
            y = (x_bwch * w).sum(dim=-1) + self.bias.view(1, 1, 1)  # (B,W,C)

        return y


class ColumnCollapseHead(BaseSpatialHead):
    """
    SpatialHead implementing the paper-style column-collapse + 1D conv pipeline.

    Pattern "B":
      - features_dim comes from the orchestrator (separate arg)
      - spec/params hold only architecture knobs
      - head always projects to features_dim

    IMPORTANT for SB3 checkpoint stability:
      - any learnable parameters must be created in __init__, not lazily in forward.
      - therefore we require in_h at construction time for the linear collapse case.
    """

    def __init__(
            self,
            *,
            in_h: int,
            in_channels: int,
            features_dim: int,
            spec: ColumnCollapseParams,
    ) -> None:
        super().__init__(features_dim=int(features_dim))
        self.spec = spec

        H = int(in_h)
        if H <= 0:
            raise ValueError(f"in_h must be > 0, got {in_h}")

        C = int(in_channels)
        if C <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

        p = float(getattr(spec, "dropout", 0.25))
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        self._collapse: CollapseKind = _normalize_collapse(spec)
        self._pool: Pool1D = _normalize_pool(spec)

        act = make_activation(getattr(spec, "activation", "relu"))
        use_bn = bool(getattr(spec, "use_batchnorm", True))

        # --- optional piece one-hot augmentation ---
        self._include_active_onehot: bool = bool(getattr(spec, "include_active_onehot", False))
        self._include_next_onehot: bool = bool(getattr(spec, "include_next_onehot", False))
        # Fixed default for classic7. If you later add other piece sets, we can extend spec cleanly.
        self._num_pieces: int = int(getattr(spec, "num_pieces", 7))
        if self._num_pieces <= 0:
            raise ValueError(f"num_pieces must be > 0, got {self._num_pieces}")

        # --- collapse module (create in __init__ for stable state_dict) ---
        self._collapse_in_h: int = H
        self._collapse_in_channels: int = C
        self._collapse_linear: Optional[_LinearHeightCollapse] = None

        if self._collapse == "linear":
            self._collapse_linear = _LinearHeightCollapse(
                H=H,
                C=C,
                per_channel=bool(getattr(spec, "linear_collapse_per_channel", True)),
            )

        # --- conv1d stack (paper defaults) ---
        chs = tuple(int(x) for x in getattr(spec, "conv_channels", (128, 128, 128)))
        ks = tuple(int(k) for k in getattr(spec, "conv_kernel_sizes", (3, 1, 3)))
        if len(chs) == 0 or len(ks) == 0 or len(chs) != len(ks):
            raise ValueError("conv_channels and conv_kernel_sizes must be non-empty and have same length")
        for k in ks:
            if k <= 0:
                raise ValueError(f"kernel sizes must be > 0, got {k}")

        layers: list[nn.Module] = []
        c_prev = C
        for c_out, k in zip(chs, ks):
            pad = (k // 2) if (k % 2 == 1) else 0
            layers.append(nn.Conv1d(c_prev, int(c_out), kernel_size=int(k), stride=1, padding=pad, bias=True))
            if use_bn:
                layers.append(nn.BatchNorm1d(int(c_out)))
            layers.append(act.__class__())
            if p > 0.0:
                layers.append(nn.Dropout(p))
            c_prev = int(c_out)

        self.conv = nn.Sequential(*layers)
        self._conv_out_channels: int = c_prev

        # --- FC stack (paper defaults: 128 -> 512) ---
        fc_layers: list[nn.Module] = []
        d_prev = self._conv_out_channels * (2 if self._pool == "avgmax" else 1)

        # include piece one-hots before FC stack
        if self._include_active_onehot:
            d_prev += self._num_pieces
        if self._include_next_onehot:
            d_prev += self._num_pieces

        fc_hidden = tuple(int(x) for x in getattr(spec, "fc_hidden", (128, 512)))
        if len(fc_hidden) == 0:
            raise ValueError("fc_hidden must be non-empty")
        post_fc_activation = bool(getattr(spec, "post_fc_activation", True))

        for i, h in enumerate(fc_hidden):
            if h <= 0:
                raise ValueError(f"fc_hidden entries must be > 0, got {h}")
            fc_layers.append(nn.Linear(d_prev, h))
            # apply activation/dropout after each FC; last one is configurable
            is_last = (i == len(fc_hidden) - 1)
            if (not is_last) or post_fc_activation:
                fc_layers.append(act.__class__())
                if p > 0.0:
                    fc_layers.append(nn.Dropout(p))
            d_prev = int(h)

        self.fc = nn.Sequential(*fc_layers)

        # Always project to features_dim.
        self.out_proj: nn.Module = (
            nn.Identity() if int(d_prev) == int(self.features_dim) else nn.Linear(d_prev, self.features_dim)
        )

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> torch.Tensor:
        x = self._check_spatial(spatial)  # (B,H,W,C)
        B, H, _W, C = x.shape

        if int(H) != int(self._collapse_in_h):
            raise ValueError(f"in_h mismatch: head expects H={self._collapse_in_h}, got H={int(H)}")
        if int(C) != int(self._collapse_in_channels):
            raise ValueError(f"in_channels mismatch: head expects C={self._collapse_in_channels}, got C={int(C)}")

        # --- collapse over height H -> (B,W,C) ---
        if self._collapse == "max":
            col = x.amax(dim=1)
        elif self._collapse == "avg":
            col = x.mean(dim=1)
        else:
            assert self._collapse_linear is not None
            col = self._collapse_linear(x)

        # --- conv1d over W ---
        col_cf = col.permute(0, 2, 1).contiguous()  # (B,C,W)
        y = self.conv(col_cf)  # (B,C',W)

        # --- pool over W ---
        if self._pool == "avg":
            pooled = y.mean(dim=2)
        elif self._pool == "max":
            pooled = y.amax(dim=2)
        else:
            m = y.mean(dim=2)
            mx = y.amax(dim=2)
            pooled = torch.cat([m, mx], dim=1)

        # --- optionally include active/next one-hots (before FC stack) ---
        if self._include_active_onehot or self._include_next_onehot:
            feats: list[torch.Tensor] = [pooled]

            if self._include_active_onehot:
                active_ids = specials.active_kind.to(device=pooled.device, dtype=torch.long)
                if active_ids.shape[0] != B:
                    raise ValueError(f"active_kind must be shape (B,), got {tuple(active_ids.shape)} for B={B}")
                feats.append(F.one_hot(active_ids, num_classes=self._num_pieces).to(pooled.dtype))

            if self._include_next_onehot:
                if specials.next_kind is None:
                    next_ids = torch.zeros((B,), device=pooled.device, dtype=torch.long)
                else:
                    next_ids = specials.next_kind.to(device=pooled.device, dtype=torch.long)
                    if next_ids.shape[0] != B:
                        raise ValueError(f"next_kind must be shape (B,), got {tuple(next_ids.shape)} for B={B}")
                feats.append(F.one_hot(next_ids, num_classes=self._num_pieces).to(pooled.dtype))

            pooled = torch.cat(feats, dim=1)

        # --- FC stack + output projection ---
        h = self.fc(pooled)
        out = self.out_proj(h)
        return self._check_out(out)


__all__ = ["ColumnCollapseHead"]
