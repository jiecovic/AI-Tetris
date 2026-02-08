# src/tetris_rl/core/game/config.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator

from tetris_rl.core.config.base import ConfigBase

PieceRule = Literal["uniform", "bag7"]


class WarmupTypeConfig(ConfigBase):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()


class WarmupConfig(ConfigBase):
    prob: float = Field(default=1.0, ge=0.0, le=1.0)
    spec: Optional[WarmupTypeConfig] = None


class GameConfig(ConfigBase):
    """
    Game-level config (engine-facing).

    Keep this as the single home for things that conceptually belong to the engine:
      - piece rule
      - warmup noise (board initialization)
      - future: scoring rulesets, gravity variants, etc.
    """

    seed: int = 12345
    piece_rule: PieceRule = "uniform"
    warmup: Optional[WarmupConfig] = None

    @field_validator("piece_rule", mode="before")
    @classmethod
    def _piece_rule_lower(cls, v: object) -> str:
        return str(v).strip().lower()


__all__ = ["GameConfig", "WarmupConfig", "WarmupTypeConfig", "PieceRule"]
