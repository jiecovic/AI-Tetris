# src/tetris_rl/core/envs/rewards/params.py
from __future__ import annotations

from typing import Mapping

from tetris_rl.core.config.base import ConfigBase


class LinesRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0


class HeuristicDeltaRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0


RewardParams = LinesRewardParams | HeuristicDeltaRewardParams

REWARD_PARAMS_REGISTRY: Mapping[str, type[ConfigBase]] = {
    "lines": LinesRewardParams,
    "heuristic_delta": HeuristicDeltaRewardParams,
}

__all__ = [
    "LinesRewardParams",
    "HeuristicDeltaRewardParams",
    "RewardParams",
    "REWARD_PARAMS_REGISTRY",
]
