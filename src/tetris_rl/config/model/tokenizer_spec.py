# src/tetris_rl/config/model/tokenizer_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

TokenizerLayout = Literal["row", "column", "patch", "row_column"]
BoardEmbedType = Literal["linear", "conv1d", "discrete_pattern"]
PaddingMode = Literal["valid", "same", "tetris"]


# ---------------------------------------------------------------------
# Layout params (YAML: tokenizer.layout.params)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RowLayoutParams:
    pass


@dataclass(frozen=True)
class ColumnLayoutParams:
    pass


@dataclass(frozen=True)
class PatchLayoutParams:
    patch_h: int = 1
    patch_w: int = 1
    stride_h: Optional[int] = None
    stride_w: Optional[int] = None


# ---------------------------------------------------------------------
# Board embedding params (YAML: tokenizer.board_embedding.params)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LinearEmbedParams:
    pass


@dataclass(frozen=True)
class DiscretePatternEmbedParams:
    pass


@dataclass(frozen=True)
class Conv1DEmbedParams:
    # preset selection
    preset: Literal["tiny", "base", "deep", "generic"] = "base"

    # shared
    dropout: float = 0.0
    padding: PaddingMode = "valid"

    # coordconv: append normalized 1D coordinate channel
    coordconv: bool = False

    # generic-only fields
    channels: tuple[int, ...] = ()
    kernel_sizes: tuple[int, ...] = ()
    strides: Optional[tuple[int, ...]] = None
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    use_batchnorm: bool = False


# ---------------------------------------------------------------------
# Typed sections (YAML: tokenizer.layout / tokenizer.board_embedding)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutConfig:
    type: TokenizerLayout
    params: RowLayoutParams | ColumnLayoutParams | PatchLayoutParams


@dataclass(frozen=True)
class BoardEmbeddingConfig:
    type: BoardEmbedType
    params: LinearEmbedParams | Conv1DEmbedParams | DiscretePatternEmbedParams


# ---------------------------------------------------------------------
# Tokenizer top-level (YAML: tokenizer)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerSpec:
    d_model: int
    layout: LayoutConfig
    board_embedding: BoardEmbeddingConfig

    add_active_token: bool = True
    add_next_token: bool = False

    share_kind_embedding: bool = True


TOKENIZER_LAYOUT_PARAMS_REGISTRY = {
    "row": RowLayoutParams,
    "column": ColumnLayoutParams,
    "patch": PatchLayoutParams,
    "row_column": RowLayoutParams,  # shares the same params container for now
}

TOKENIZER_BOARD_EMBED_PARAMS_REGISTRY = {
    "linear": LinearEmbedParams,
    "conv1d": Conv1DEmbedParams,
    "discrete_pattern": DiscretePatternEmbedParams,
}

__all__ = [
    "TokenizerLayout",
    "BoardEmbedType",
    "RowLayoutParams",
    "ColumnLayoutParams",
    "PatchLayoutParams",
    "LinearEmbedParams",
    "DiscretePatternEmbedParams",
    "Conv1DEmbedParams",
    "LayoutConfig",
    "BoardEmbeddingConfig",
    "TokenizerSpec",
]
