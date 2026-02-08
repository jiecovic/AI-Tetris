# src/tetris_rl/core/envs/rewards/lines_survival.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import LinesSurvivalRewardParams


class LinesSurvivalReward(RewardFn):
    """
    Lines-cleared reward with per-step survival bonus, plus penalties.
    """

    def __init__(self, *, spec: LinesSurvivalRewardParams) -> None:
        self.illegal_penalty = float(spec.illegal_penalty)
        self.terminal_penalty = 0.0
        self.survival_bonus = float(spec.survival_bonus)

    def __call__(
        self,
        *,
        prev_state: Any,
        action: Any,
        next_state: Any,
        features: TransitionFeatures,
        info: Dict[str, Any],
    ) -> float:
        _ = prev_state
        _ = action
        _ = next_state
        _ = info

        r = 0.0

        if bool(getattr(features, "invalid_action", False)):
            r -= float(self.illegal_penalty)
            if bool(getattr(features, "game_over", False)):
                r -= float(self.terminal_penalty)
            return float(r)

        r += float(getattr(features, "cleared_lines", 0) or 0)
        if not bool(getattr(features, "game_over", False)):
            r += float(self.survival_bonus)

        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)


__all__ = ["LinesSurvivalReward"]
