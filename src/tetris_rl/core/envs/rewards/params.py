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


class LinesSurvivalRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.1


RewardParams = LinesRewardParams | HeuristicDeltaRewardParams | LinesSurvivalRewardParams

REWARD_PARAMS_REGISTRY: Mapping[str, type[ConfigBase]] = {
    "lines": LinesRewardParams,
    "heuristic_delta": HeuristicDeltaRewardParams,
    "lines_survival": LinesSurvivalRewardParams,
}

__all__ = [
    "LinesRewardParams",
    "HeuristicDeltaRewardParams",
    "LinesSurvivalRewardParams",
    "RewardParams",
    "REWARD_PARAMS_REGISTRY",
]
