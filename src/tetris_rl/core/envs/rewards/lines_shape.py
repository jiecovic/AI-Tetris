# src/tetris_rl/core/envs/rewards/lines_shape.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import LinesShapeRewardParams


class LinesShapeReward(RewardFn):
    """
    Reward shaping from board deltas and special line clears.

    Rules (params control magnitudes):
      - +line_cleared_bonus if cleared_lines > 0.
      - -hole_added_penalty if delta_holes > 0.
      - +no_new_holes_bonus if delta_holes <= 0.
      - +hole_removed_bonus if delta_holes < 0.
      - +tetris_bonus if cleared_lines == 4.
    """

    def __init__(self, *, spec: LinesShapeRewardParams) -> None:
        self.illegal_penalty = float(spec.illegal_penalty)
        self.terminal_penalty = float(spec.terminal_penalty)
        self.survival_bonus = float(spec.survival_bonus)
        self.line_cleared_bonus = float(spec.line_cleared_bonus)
        self.hole_added_penalty = float(spec.hole_added_penalty)
        self.no_new_holes_bonus = float(spec.no_new_holes_bonus)
        self.hole_removed_bonus = float(spec.hole_removed_bonus)
        self.tetris_bonus = float(spec.tetris_bonus)

    def __call__(
        self,
        *,
        prev_state: Any,
        action: Any,
        next_state: Any,
        features: TransitionFeatures,
        info: Dict[str, Any],
    ) -> float:
        r = 0.0

        invalid = bool(getattr(features, "invalid_action", False))
        game_over = bool(getattr(features, "game_over", False))

        if invalid:
            r -= float(self.illegal_penalty)
            if game_over:
                r -= float(self.terminal_penalty)
            return float(r)

        d_holes = float(getattr(features, "delta_holes", 0) or 0)

        cleared = int(getattr(features, "cleared_lines", 0) or 0)
        cleared = max(0, min(cleared, 4))

        if cleared > 0:
            r += float(self.line_cleared_bonus)

        if d_holes > 0:
            r -= float(self.hole_added_penalty)
        else:
            r += float(self.no_new_holes_bonus)
            if d_holes < 0:
                r += float(self.hole_removed_bonus)

        if cleared == 4:
            r += float(self.tetris_bonus)

        if not game_over:
            r += float(self.survival_bonus)

        if game_over:
            r -= float(self.terminal_penalty)

        return float(r)


__all__ = ["LinesShapeReward"]
