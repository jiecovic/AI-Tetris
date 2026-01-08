# src/tetris_rl/envs/catalog.py
from __future__ import annotations

from typing import Any, Mapping

from tetris_rl.envs.macro_env import MacroTetrisEnv
from tetris_rl.envs.rewards.holes import HolesDeltaReward
from tetris_rl.envs.rewards.heuristic_delta import HeuristicDeltaReward
from tetris_rl.envs.rewards.heuristic_softplus import HeuristicSoftplusReward
from tetris_rl.envs.rewards.lines import LinesReward
from tetris_rl.envs.rewards.learned_ridge_delta import LearnedRidgeDeltaReward
from tetris_rl.envs.rewards.shaped import ShapedMacroReward
from tetris_rl.envs.rewards.sparse_reward import SparseReward
from tetris_rl.envs.rewards.heuristic_delta_piecewise import HeuristicDeltaPiecewiseReward
from tetris_rl.envs.rewards.heuristic_linear import HeuristicLinear


# - imports + plain dicts only
# - no functions/classes
# - easy to add new entries

ENV_REGISTRY: Mapping[str, Any] = {
    "macro": MacroTetrisEnv,
}

REWARD_REGISTRY: Mapping[str, Any] = {
    "lines": LinesReward,
    "shaped": ShapedMacroReward,
    "heuristic_delta": HeuristicDeltaReward,
    "heuristic_delta_piecewise": HeuristicDeltaPiecewiseReward,
    "heuristic_softplus": HeuristicSoftplusReward,
    "holes": HolesDeltaReward,
    "learned_ridge": LearnedRidgeDeltaReward,
    "heuristic_linear": HeuristicLinear,
    "sparse_reward": SparseReward,
}
