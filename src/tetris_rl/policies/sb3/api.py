# src/tetris_rl/policies/sb3/api.py
from __future__ import annotations

"""
Global model-side API contracts.

This module defines the *stable interfaces and data carriers* that all
model components agree on. These are NOT implementations.

Design principles:
- Keep these minimal and boring.
- No architecture decisions here.
- Used across spatial preprocessors, tokenizers, mixers, and heads.
- Safe to import everywhere.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class Specials:
    """
    Global (non-spatial) conditioning signals from the environment.

    These are *not* part of the spatial grid and may be injected via:
      - token stream (special tokens)
      - FiLM conditioning
      - bypass / concat to feature vector

    Shapes:
      active_kind: (B,) int64
      next_kind:   (B,) int64 or None
    """
    active_kind: torch.Tensor
    next_kind: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SpatialFeatures:
    """
    Generic spatial feature grid flowing through the model.

    This represents ANY grid-like intermediate:
      - binary board (C=1)
      - RGB board   (C=3)
      - CNN stem output (C>1, H/W possibly changed)

    Invariants:
      - channel-last layout: (B, H, W, C)
      - float32 tensor
      - no tokenization or pooling applied yet

    `is_discrete_binary` indicates whether x represents *exact* {0,1}
    occupancy before any learned transformation. This is used to
    gate LUT-based token embeddings.
    """
    x: torch.Tensor  # (B, H, W, C) float32
    is_discrete_binary: bool = False


@dataclass(frozen=True)
class TokenStream:
    """
    Single concatenated token stream consumed by a TokenMixer.

    This stream may contain heterogeneous tokens:
      - row tokens
      - column tokens
      - patch tokens
      - special tokens
      - CLS tokens

    Shapes:
      x:     (B, T, D) float32 token embeddings
      types: (T,) int64 token-type ids aligned with x positions

    NOTE:
    `extra` (bypass features) is intentionally NOT part of this contract.
    If you need to inject Specials after mixing, do it via FeatureAugmenter
    on the final (B,F) feature vector.
    """
    x: torch.Tensor
    types: torch.Tensor


# ---------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------

@runtime_checkable
class SpatialPreprocessor(Protocol):
    """
    Converts raw observations into SpatialFeatures and Specials.

    Typical responsibilities:
      - parse obs dict
      - convert categorical board -> binary / RGB
      - apply optional CNN stem
      - apply optional FiLM conditioning

    Output MUST still be spatial.
    """

    def __call__(
            self,
            *,
            observations: dict[str, torch.Tensor],
    ) -> tuple[SpatialFeatures, Specials]: ...

    def out_spec(self, *, board: BoardSpec) -> SpatialSpec: ...


@runtime_checkable
class Tokenizer(Protocol):
    """
    Converts SpatialFeatures (+ Specials) into a TokenStream.

    Responsibilities:
      - extract row / col / patch tokens (possibly multiple at once)
      - embed tokens to D
      - assemble token-type ids
      - optionally add special / CLS tokens

    Output is no longer spatial.
    """

    def __call__(
            self,
            *,
            spatial: SpatialFeatures,
            specials: Specials,
    ) -> TokenStream: ...

    def stream_spec(self) -> TokenStreamSpec: ...

@runtime_checkable
class TokenMixer(Protocol):
    """
    Consumes a TokenStream and produces a fixed-size feature vector.

    Examples:
      - MLP mixer
      - Transformer / ViT mixer
    """

    def __call__(self, *, stream: TokenStream) -> torch.Tensor: ...


@runtime_checkable
class SpatialHead(Protocol):
    """
    Spatial path: SpatialFeatures (+ Specials) -> feature vector.

    Used when no tokenization is desired.
    """

    def __call__(
            self,
            *,
            spatial: SpatialFeatures,
            specials: Specials,
    ) -> torch.Tensor: ...


@runtime_checkable
class FeatureAugmenter(Protocol):
    """
    Feature-level post-processing:

      base_features: (B,F)
      specials:      Specials
      -> out_features: (B,F_out)

    Used to inject Specials into the final feature vector (concat, MLP, etc.)
    after either:
      - token route (tokenizer -> token_mixer)
      - spatial route (spatial_head)

    This module owns *how* specials are represented (one-hot, embedding, MLP, etc.).
    """

    def __call__(
            self,
            *,
            features: torch.Tensor,
            specials: Specials,
    ) -> torch.Tensor: ...


@runtime_checkable
class SpatialStem(Protocol):
    def out_spec(self, *, in_spec: SpatialSpec) -> SpatialSpec: ...

    def __call__(self, spatial: SpatialFeatures) -> SpatialFeatures: ...


@dataclass(frozen=True)
class BoardSpec:
    h: int
    w: int


@dataclass(frozen=True)
class SpatialSpec:
    h: int
    w: int
    c: int
    is_discrete_binary: bool = False


@dataclass(frozen=True)
class TokenStreamSpec:
    T: int
    d_model: int


__all__ = [
    "Specials",
    "SpatialFeatures",
    "TokenStream",
    "SpatialPreprocessor",
    "Tokenizer",
    "TokenMixer",
    "SpatialHead",
    "FeatureAugmenter",
]

