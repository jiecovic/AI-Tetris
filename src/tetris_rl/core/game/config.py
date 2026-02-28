# src/tetris_rl/core/game/config.py
from __future__ import annotations

from typing import Literal, Mapping, Optional

from pydantic import Field, field_validator, model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.game.warmup_spec import WarmupSpec

PieceRule = Literal["uniform", "bag7"]


def _as_int(value: object, *, where: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be an int, got bool")
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        raise TypeError(f"{where} must be an int-like value, got {type(value)!r}")
    try:
        return int(value)
    except Exception as e:
        raise TypeError(f"{where} must be an int-like value, got {type(value)!r}") from e


class WarmupConfig(ConfigBase):
    prob: float = Field(default=1.0, ge=0.0, le=1.0)
    spec: Optional[WarmupSpec] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_spec_type(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        spec = out.get("spec", None)
        if isinstance(spec, Mapping):
            spec_out = dict(spec)
            if "type" in spec_out:
                spec_out["type"] = str(spec_out["type"]).strip().lower()
            out["spec"] = spec_out
        return out

    @model_validator(mode="before")
    @classmethod
    def _require_spec_when_enabled(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        prob_raw = out.get("prob", 1.0)
        try:
            prob = float(prob_raw)
        except Exception as e:
            raise TypeError(f"warmup.prob must be numeric, got {type(prob_raw)!r}") from e
        if prob > 0.0 and out.get("spec", None) is None:
            raise ValueError("warmup.spec is required when warmup.prob > 0")
        return out


class GameConfig(ConfigBase):
    """
    Game-level config (engine-facing).

    Keep this as the single home for things that conceptually belong to the engine:
      - piece rule
      - warmup noise (board initialization)
      - future: scoring rulesets, gravity variants, etc.
    """

    seed: int = Field(default=12345, ge=0)
    piece_rule: PieceRule = "uniform"
    warmup: Optional[WarmupConfig] = None

    @field_validator("seed", mode="before")
    @classmethod
    def _seed_int(cls, v: object) -> int:
        return _as_int(v, where="game.seed")

    @field_validator("piece_rule", mode="before")
    @classmethod
    def _piece_rule_lower(cls, v: object) -> str:
        return str(v).strip().lower()


__all__ = ["GameConfig", "WarmupConfig", "PieceRule"]
