# src/tetris_rl/env_bundles/rewards/heuristic_delta.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
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

    # === CodemyRoad weights (magnitudes for penalty terms) ===
    # b (Complete Lines) is positive
    w_lines: float = 0.760666

    # a, c, d are negative in the original score; we store magnitudes and subtract
    w_holes: float = -0.35663
    w_bumpiness: float = -0.184483
    w_agg_height: float = -0.510066

    # Not part of CodemyRoad's 4-feature score (keep for interface compatibility)
    w_max_height: float = 0.0

    # CodemyRoad score has no constant bias (scale-invariant comparison)
    survival_bonus: float = 0.0

    # Penalties (POSITIVE magnitudes)
    illegal_penalty: float = 10.0
    terminal_penalty: float = 10.0

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
