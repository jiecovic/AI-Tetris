from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tetris_rl.config.base import ConfigBase
from tetris_rl.config.typed_params import parse_typed_params

TokenizerLayout = Literal["row", "column", "patch", "row_column"]
BoardEmbedType = Literal["linear", "conv1d", "discrete_pattern"]
PaddingMode = Literal["valid", "same", "tetris"]


# ---------------------------------------------------------------------
# Layout params (YAML: tokenizer.layout.params)
# ---------------------------------------------------------------------


class RowLayoutParams(ConfigBase):
    pass


class ColumnLayoutParams(ConfigBase):
    pass


class PatchLayoutParams(ConfigBase):
    patch_h: int = 1
    patch_w: int = 1
    stride_h: int | None = None
    stride_w: int | None = None


# ---------------------------------------------------------------------
# Board embedding params (YAML: tokenizer.board_embedding.params)
# ---------------------------------------------------------------------


class LinearEmbedParams(ConfigBase):
    pass


class DiscretePatternEmbedParams(ConfigBase):
    pass


class Conv1DEmbedParams(ConfigBase):
    # preset selection
    preset: Literal["tiny", "base", "deep", "deep_l5", "generic"] = "base"

    # shared
    dropout: float = 0.0
    padding: PaddingMode = "valid"

    # coordconv: append normalized 1D coordinate channel
    coordconv: bool = False

    # generic-only fields
    channels: tuple[int, ...] = ()
    kernel_sizes: tuple[int, ...] = ()
    strides: tuple[int, ...] | None = None
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    use_batchnorm: bool = False


# ---------------------------------------------------------------------
# Typed sections (YAML: tokenizer.layout / tokenizer.board_embedding)
# ---------------------------------------------------------------------


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


class LayoutConfig(ConfigBase):
    type: TokenizerLayout
    params: RowLayoutParams | ColumnLayoutParams | PatchLayoutParams

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, LayoutConfig):
            return data
        if not isinstance(data, dict):
            raise TypeError("tokenizer.layout must be a mapping with keys {type, params}")
        tag, params = parse_typed_params(
            type_value=data.get("type", None),
            params_value=data.get("params", None),
            registry=TOKENIZER_LAYOUT_PARAMS_REGISTRY,
            where="tokenizer.layout",
        )
        return {"type": tag, "params": params}


class BoardEmbeddingConfig(ConfigBase):
    type: BoardEmbedType
    params: LinearEmbedParams | Conv1DEmbedParams | DiscretePatternEmbedParams

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, BoardEmbeddingConfig):
            return data
        if not isinstance(data, dict):
            raise TypeError("tokenizer.board_embedding must be a mapping with keys {type, params}")
        tag, params = parse_typed_params(
            type_value=data.get("type", None),
            params_value=data.get("params", None),
            registry=TOKENIZER_BOARD_EMBED_PARAMS_REGISTRY,
            where="tokenizer.board_embedding",
        )
        return {"type": tag, "params": params}


# ---------------------------------------------------------------------
# Tokenizer top-level (YAML: tokenizer)
# ---------------------------------------------------------------------


class TokenizerSpec(ConfigBase):
    d_model: int = Field(ge=1)
    layout: LayoutConfig
    board_embedding: BoardEmbeddingConfig

    add_active_token: bool = True
    add_next_token: bool = False

    share_kind_embedding: bool = True


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
    "TOKENIZER_LAYOUT_PARAMS_REGISTRY",
    "TOKENIZER_BOARD_EMBED_PARAMS_REGISTRY",
]

