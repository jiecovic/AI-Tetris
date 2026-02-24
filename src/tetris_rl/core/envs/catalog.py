# src/tetris_rl/core/envs/catalog.py
from __future__ import annotations

from typing import Any, Mapping

from tetris_rl.core.envs.macro_env import MacroTetrisEnv
from tetris_rl.core.envs.rewards.heuristic_delta import HeuristicDeltaReward
from tetris_rl.core.envs.rewards.lines import LinesReward
from tetris_rl.core.envs.rewards.lines_clean import LinesCleanReward
from tetris_rl.core.envs.rewards.lines_height_scaled import LinesHeightScaledReward
from tetris_rl.core.envs.rewards.lines_shape import LinesShapeReward

# - imports + plain dicts only
# - no functions/classes
# - easy to add new entries

ENV_REGISTRY: Mapping[str, Any] = {
    "macro": MacroTetrisEnv,
}

REWARD_REGISTRY: Mapping[str, Any] = {
    "lines": LinesReward,
    "heuristic_delta": HeuristicDeltaReward,
    "lines_clean": LinesCleanReward,
    "lines_height_scaled": LinesHeightScaledReward,
    "lines_shape": LinesShapeReward,
}
