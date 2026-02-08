# src/tetris_rl/core/envs/rewards/params.py
from __future__ import annotations

from typing import Mapping

from tetris_rl.core.config.base import ConfigBase


class LinesRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0


class HeuristicDeltaRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0
    w_lines: float = 0.760666
    w_holes: float = -0.35663
    w_bumpiness: float = -0.184483
    w_agg_height: float = -0.510066
    survival_bonus: float = 0.0


class LinesCleanRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0
    no_new_holes_bonus: float = 1.0


RewardParams = LinesRewardParams | HeuristicDeltaRewardParams | LinesCleanRewardParams

REWARD_PARAMS_REGISTRY: Mapping[str, type[ConfigBase]] = {
    "lines": LinesRewardParams,
    "heuristic_delta": HeuristicDeltaRewardParams,
    "lines_clean": LinesCleanRewardParams,
}

__all__ = [
    "LinesRewardParams",
    "HeuristicDeltaRewardParams",
    "LinesCleanRewardParams",
    "RewardParams",
    "REWARD_PARAMS_REGISTRY",
]
