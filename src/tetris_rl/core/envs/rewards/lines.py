# src/tetris_rl/core/envs/rewards/lines.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import LinesRewardParams


class LinesReward(RewardFn):
    """
    Baseline reward: number of lines cleared on the placement, plus penalties.
    """

    def __init__(self, *, spec: LinesRewardParams) -> None:
        self.illegal_penalty = float(spec.illegal_penalty)
        self.terminal_penalty = float(spec.terminal_penalty)

    def __call__(
            self,
            *,
            prev_state: Any,
            action: Any,
            next_state: Any,
            features: TransitionFeatures,
            info: Dict[str, Any],
    ) -> float:
        r = float(getattr(features, "cleared_lines", 0) or 0)

        if bool(getattr(features, "invalid_action", False)):
            r -= float(self.illegal_penalty)

        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
