# src/tetris_rl/env_bundles/catalog.py
from __future__ import annotations

from typing import Any, Mapping

from tetris_rl.envs.macro_env import MacroTetrisEnv
from tetris_rl.envs.rewards.heuristic_delta import HeuristicDeltaReward
from tetris_rl.envs.rewards.lines import LinesReward
from tetris_rl.envs.rewards.params import (
    HeuristicDeltaRewardParams,
    LinesRewardParams,
)


# - imports + plain dicts only
# - no functions/classes
# - easy to add new entries

ENV_REGISTRY: Mapping[str, Any] = {
    "macro": MacroTetrisEnv,
}

REWARD_REGISTRY: Mapping[str, Any] = {
    "lines": LinesReward,
    "heuristic_delta": HeuristicDeltaReward,
}

REWARD_PARAMS_REGISTRY: Mapping[str, Any] = {
    "lines": LinesRewardParams,
    "heuristic_delta": HeuristicDeltaRewardParams,
}
