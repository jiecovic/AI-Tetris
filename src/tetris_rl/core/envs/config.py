# src/tetris_rl/core/envs/config.py
from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field, model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.config.typed_params import parse_typed_params
from tetris_rl.core.envs.invalid_action import InvalidActionPolicy
from tetris_rl.core.envs.macro_actions import ActionMode
from tetris_rl.core.envs.rewards.params import REWARD_PARAMS_REGISTRY, RewardParams
from tetris_rl.core.game.config import GameConfig


class MacroEnvParams(ConfigBase):
    action_mode: ActionMode = "discrete"
    max_steps: int | None = None
    invalid_action_policy: InvalidActionPolicy = "noop"


class EnvConfig(ConfigBase):
    type: str
    params: MacroEnvParams
    reward: "RewardConfig"
    game: GameConfig

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, EnvConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("env must be a mapping with keys {type, params, reward, game}")
        tag, params = parse_typed_params(
            type_value=data.get("type", None),
            params_value=data.get("params", None),
            registry={"macro": MacroEnvParams},
            where="env",
        )
        out = dict(data)
        out["type"] = tag
        out["params"] = params
        return out


class RewardConfig(ConfigBase):
    type: str
    params: RewardParams = Field(default_factory=lambda: REWARD_PARAMS_REGISTRY["lines"]())

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, RewardConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError("env.reward must be a mapping with keys {type, params}")
        tag, params = parse_typed_params(
            type_value=data.get("type", None),
            params_value=data.get("params", None),
            registry=REWARD_PARAMS_REGISTRY,
            where="env.reward",
        )
        return {"type": tag, "params": params}


__all__ = ["EnvConfig", "RewardConfig", "MacroEnvParams"]
