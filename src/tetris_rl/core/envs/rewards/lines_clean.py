# src/tetris_rl/core/envs/rewards/lines_clean.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import LinesCleanRewardParams


class LinesCleanReward(RewardFn):
    """
    Reward for cleared lines, plus a bonus when no new holes are added.
    Optional survival bonus can be added for non-terminal steps.
    """

    def __init__(self, *, spec: LinesCleanRewardParams) -> None:
        self.invalid_penalty = float(spec.invalid_penalty)
        self.terminal_penalty = float(spec.terminal_penalty)
        self.survival_bonus = float(spec.survival_bonus)
        self.no_new_holes_bonus = float(spec.no_new_holes_bonus)

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

        invalid = bool(getattr(features, "invalid_action", False))
        game_over = bool(getattr(features, "game_over", False))

        if invalid:
            r -= float(self.invalid_penalty)
            if game_over:
                r -= float(self.terminal_penalty)
            return float(r)

        new_holes = float(getattr(features, "delta_holes", 0) or 0)
        if new_holes <= 0:
            r += float(self.no_new_holes_bonus)

        if not game_over:
            r += float(self.survival_bonus)

        if game_over:
            r -= float(self.terminal_penalty)

        return float(r)


__all__ = ["LinesCleanReward"]
