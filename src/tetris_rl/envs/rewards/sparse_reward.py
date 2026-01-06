# src/tetris_rl/envs/rewards/sparse_reward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
class SparseReward(RewardFn):
    """
    Simple, sparse, placement-level reward.

    Terms (applied only for legal actions):
      - +line_reward * cleared_lines
      - +no_new_holes_bonus if delta_holes == 0
      - +hole_reduction_reward_per_hole * (-delta_holes) if delta_holes < 0
      - -new_hole_penalty_per_hole * delta_holes if delta_holes > 0
      - +no_height_increase_bonus if delta_agg_height <= 0

    Illegal actions:
      - Subtract illegal_penalty.
      - Terminal penalty is still subtracted if game_over is True.
    """

    # sparse positives
    line_reward: float = 1.0
    no_new_holes_bonus: float = 1.0
    no_height_increase_bonus: float = 1.0
    hole_reduction_reward_per_hole: float = 1.0

    # sparse negatives
    new_hole_penalty_per_hole: float = 1.0

    # penalties (positive magnitudes)
    illegal_penalty: float = 50.0
    terminal_penalty: float = 50.0

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

        # ------------------------------------------------------------
        # Illegal action: penalties only
        # ------------------------------------------------------------
        if bool(getattr(features, "illegal_action", False)):
            r -= float(self.illegal_penalty)
            if bool(getattr(features, "game_over", False)):
                r -= float(self.terminal_penalty)
            return float(r)

        # cleared lines (robust cap)
        cl = int(getattr(features, "cleared_lines", 0) or 0)
        cl = max(0, min(cl, 4))

        # deltas (missing -> 0.0)
        dh = float(getattr(features, "delta_holes", 0) or 0)
        dah = float(getattr(features, "delta_agg_height", 0) or 0)

        # ------------------------------------------------------------
        # Sparse reward terms
        # ------------------------------------------------------------
        # +1 per cleared line
        r += float(self.line_reward) * float(cl)

        # holes logic
        if dh == 0.0:
            # exactly no new holes
            r += float(self.no_new_holes_bonus)
        elif dh < 0.0:
            # reward hole reduction (scaled)
            r += float(self.hole_reduction_reward_per_hole) * float(-dh)
        else:
            # penalize new holes
            r -= 2

        # +1 if aggregate height did not increase
        if dah <= 0.0:
            r += 0.5*float(self.no_height_increase_bonus)
        else:
            r -= 1

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
