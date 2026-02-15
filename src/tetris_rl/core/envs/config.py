# src/tetris_rl/core/envs/config.py
from __future__ import annotations

from typing import Literal, Mapping, cast

from pydantic import Field, field_validator, model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.config.typed_params import parse_typed_params
from tetris_rl.core.envs.actions import ActionMode, InvalidActionPolicy
from tetris_rl.core.envs.rewards.params import REWARD_PARAMS_REGISTRY, RewardParams
from tetris_rl.core.game.config import GameConfig

InfoLevel = Literal["train", "watch"]


class MacroEnvParams(ConfigBase):
    action_mode: ActionMode = "discrete"
    max_steps: int | None = None
    invalid_action_policy: InvalidActionPolicy = "noop"
    feature_clear_mode: str = "post"
    info_level: InfoLevel = "train"

    @model_validator(mode="before")
    @classmethod
    def _normalize_feature_clear_mode(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        raw = data.get("feature_clear_mode", "post")
        mode = str(raw).strip().lower()
        if mode in {"lock", "pre", "pre_clear", "before"}:
            mode = "lock"
        elif mode in {"post", "clear", "post_clear", "after"}:
            mode = "post"
        else:
            raise ValueError("env.params.feature_clear_mode must be pre|lock|post|clear")
        out = dict(data)
        out["feature_clear_mode"] = mode
        return out

    @field_validator("info_level", mode="before")
    @classmethod
    def _normalize_info_level(cls, v: object) -> str:
        s = str(v).strip().lower()
        if s in {"watch", "ui", "full", "debug"}:
            return "watch"
        if s in {"train", "min", "minimal"}:
            return "train"
        raise ValueError("env.params.info_level must be 'train' or 'watch'")


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
    params: RewardParams = Field(default_factory=lambda: cast(RewardParams, REWARD_PARAMS_REGISTRY["lines"]()))

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
        return {"type": tag, "params": cast(RewardParams, params)}


__all__ = ["EnvConfig", "RewardConfig", "MacroEnvParams"]
