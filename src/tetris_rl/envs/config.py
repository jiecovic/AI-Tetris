from __future__ import annotations

from typing import Any, Dict

from pydantic import Field, field_validator

from tetris_rl.config.base import ConfigBase
from tetris_rl.game.config import GameConfig


class RewardConfig(ConfigBase):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()


class EnvConfig(ConfigBase):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reward: RewardConfig
    game: GameConfig

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()


__all__ = ["EnvConfig", "RewardConfig"]
