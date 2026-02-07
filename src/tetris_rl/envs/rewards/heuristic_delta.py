# src/tetris_rl/env_bundles/rewards/heuristic_delta.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures
from tetris_rl.envs.rewards.params import HeuristicDeltaRewardParams


class HeuristicDeltaReward(RewardFn):
    """
    Placement-level reward shaping using CodemyRoad's classic 4-feature heuristic
    as a delta reward (score(next) - score(prev)).

    CodemyRoad "Near Perfect Bot" score:
      score = a*agg_height + b*complete_lines + c*holes + d*bumpiness
      a = -0.510066
      b = +0.760666
      c = -0.35663
      d = -0.184483

    Delta shaping:
      r ≈ b * Δlines - |c| * Δholes - |a| * Δagg_height - |d| * Δbumpiness

    Conventions:
      - All penalties are POSITIVE magnitudes.
      - Penalties are SUBTRACTED at computation time.

    Illegal actions:
      - Subtract illegal_penalty.
      - Do NOT receive learned terms (no bias, no deltas).
      - Terminal penalty is still subtracted if game_over is True.
    """

    # CodemyRoad weights (fixed constants).
    _W_LINES = 0.760666
    _W_HOLES = -0.35663
    _W_BUMPINESS = -0.184483
    _W_AGG_HEIGHT = -0.510066
    _SURVIVAL_BONUS = 0.0

    # Penalties (POSITIVE magnitudes).
    illegal_penalty: float
    terminal_penalty: float

    def __init__(self, *, spec: HeuristicDeltaRewardParams) -> None:
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
        r = 0.0

        # ------------------------------------------------------------
        # Illegal action: penalties only
        # ------------------------------------------------------------
        if bool(getattr(features, "invalid_action", False)):
            r -= float(self.illegal_penalty)
            if bool(getattr(features, "game_over", False)):
                r -= float(self.terminal_penalty)
            return float(r)

        # cleared lines (robust cap)
        cl = int(getattr(features, "cleared_lines", 0) or 0)
        cl = max(0, min(cl, 4))

        # deltas (missing -> 0.0)
        dh = float(getattr(features, "delta_holes", 0) or 0)
        db = float(getattr(features, "delta_bumpiness", 0) or 0)
        dah = float(getattr(features, "delta_agg_height", 0) or 0)

        # max height delta is unused here (CodemyRoad doesn't include it)
        # dmh = float(getattr(features, "delta_max_height", 0) or 0)

        # CodemyRoad delta-shaped reward
        r += float(self._W_LINES) * float(cl)
        r += float(self._W_HOLES) * float(dh)
        r += float(self._W_BUMPINESS) * float(db)
        r += float(self._W_AGG_HEIGHT) * float(dah)

        # bias (kept for compatibility; default 0.0)
        r += float(self._SURVIVAL_BONUS)

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
