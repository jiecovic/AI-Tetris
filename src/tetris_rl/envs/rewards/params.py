# src/tetris_rl/envs/rewards/params.py
from __future__ import annotations

from tetris_rl.config.base import ConfigBase


class LinesRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0


class HeuristicDeltaRewardParams(ConfigBase):
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0


RewardParams = LinesRewardParams | HeuristicDeltaRewardParams

__all__ = [
    "LinesRewardParams",
    "HeuristicDeltaRewardParams",
    "RewardParams",
]
