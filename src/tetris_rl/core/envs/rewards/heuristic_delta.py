# src/tetris_rl/core/envs/rewards/heuristic_delta.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import HeuristicDeltaRewardParams


class HeuristicDeltaReward(RewardFn):
    """
    Placement-level reward shaping using a 4-feature heuristic
    as a delta reward (score(next) - score(prev)).

    CodemyRoad "Near Perfect Bot" score (default params):
      https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
      score = a*agg_height + b*complete_lines + c*holes + d*bumpiness
      a = -0.510066
      b = +0.760666
      c = -0.35663
      d = -0.184483

    Delta shaping (a,b,c,d order):
      r ~= a * d_agg_height + b * d_lines + c * d_holes + d * d_bumpiness
      (a,c,d negative; b positive for Codemy defaults)

    Feature source:
      - Uses env-selected deltas (lock/post) from TransitionFeatures.

    Params are tunable via YAML; use the codemy preset to lock defaults.

    Conventions:
      - All penalties are POSITIVE magnitudes.
      - Penalties are SUBTRACTED at computation time.

    Invalid actions:
      - Subtract invalid_penalty.
      - Do NOT receive learned terms (no bias, no deltas).
      - Terminal penalty is still subtracted if game_over is True.
    """

    def __init__(self, *, spec: HeuristicDeltaRewardParams) -> None:
        self.invalid_penalty = float(spec.invalid_penalty)
        self.terminal_penalty = float(spec.terminal_penalty)
        self.w_lines = float(spec.w_lines)
        self.w_holes = float(spec.w_holes)
        self.w_bumpiness = float(spec.w_bumpiness)
        self.w_agg_height = float(spec.w_agg_height)
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
        r = 0.0

        # ------------------------------------------------------------
        # Invalid action: penalties only
        # ------------------------------------------------------------
        if bool(getattr(features, "invalid_action", False)):
            r -= float(self.invalid_penalty)
            if bool(getattr(features, "game_over", False)):
                r -= float(self.terminal_penalty)
            return float(r)

        # cleared lines (robust cap)
        cl = int(getattr(features, "cleared_lines", 0) or 0)
        cl = max(0, min(cl, 4))

        # deltas (from env-selected feature_clear_mode)
        dh = float(getattr(features, "delta_holes", 0) or 0)
        db = float(getattr(features, "delta_bumpiness", 0) or 0)
        dah = float(getattr(features, "delta_agg_height", 0) or 0)

        # max height delta is unused here (CodemyRoad doesn't include it)
        # dmh = float(getattr(features, "delta_max_height", 0) or 0)

        # CodemyRoad delta-shaped reward
        r += float(self.w_lines) * float(cl)
        r += float(self.w_holes) * float(dh)
        r += float(self.w_bumpiness) * float(db)
        r += float(self.w_agg_height) * float(dah)

        # bias (kept for compatibility; default 0.0)
        r += float(self.survival_bonus)

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
