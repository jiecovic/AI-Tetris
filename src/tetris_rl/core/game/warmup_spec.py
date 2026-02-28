# src/tetris_rl/core/game/warmup_spec.py
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.game.warmup_params import (
    WarmupBasePlusPoissonParams,
    WarmupFixedParams,
    WarmupNoneParams,
    WarmupPoissonParams,
    WarmupUniformRowsParams,
)

WarmupSpecType = Literal[
    "none",
    "off",
    "disabled",
    "null",
    "fixed",
    "uniform_rows",
    "uniform",
    "poisson",
    "base_plus_poisson",
    "base+poisson",
]


class WarmupNoneSpec(ConfigBase):
    type: Literal["none", "off", "disabled", "null"] = "none"
    params: WarmupNoneParams = Field(default_factory=WarmupNoneParams)


class WarmupFixedSpec(ConfigBase):
    type: Literal["fixed"] = "fixed"
    params: WarmupFixedParams


class WarmupUniformRowsSpec(ConfigBase):
    type: Literal["uniform_rows", "uniform"] = "uniform_rows"
    params: WarmupUniformRowsParams


class WarmupPoissonSpec(ConfigBase):
    type: Literal["poisson"] = "poisson"
    params: WarmupPoissonParams


class WarmupBasePlusPoissonSpec(ConfigBase):
    type: Literal["base_plus_poisson", "base+poisson"] = "base_plus_poisson"
    params: WarmupBasePlusPoissonParams


WarmupSpec = Annotated[
    WarmupNoneSpec | WarmupFixedSpec | WarmupUniformRowsSpec | WarmupPoissonSpec | WarmupBasePlusPoissonSpec,
    Field(discriminator="type"),
]


__all__ = [
    "WarmupSpecType",
    "WarmupNoneSpec",
    "WarmupFixedSpec",
    "WarmupUniformRowsSpec",
    "WarmupPoissonSpec",
    "WarmupBasePlusPoissonSpec",
    "WarmupSpec",
]
