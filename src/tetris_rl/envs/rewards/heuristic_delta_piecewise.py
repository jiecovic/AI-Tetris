# src/tetris_rl/env_bundles/rewards/heuristic_delta_piecewise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
class HeuristicDeltaPiecewiseReward(RewardFn):
    """
    Placement-level reward shaping using a learned piecewise-linear reward over delta-features.

    Model:
      r = bias
          + sum_i w_lin[i] * x_i
          + sum_i sum_k w_hinge[i,k] * max(0, x_i - knot[i,k])

    Notes:
      - We keep the implementation explicit and cheap (a few max(0, x) hinges).

    Conventions:
      - All penalties are POSITIVE magnitudes.
      - Penalties are SUBTRACTED at computation time.

    Illegal actions:
      - Subtract illegal_penalty.
      - Do NOT receive learned terms (no bias, no deltas).
      - Terminal penalty is still subtracted if game_over is True.
    """

    # ------------------------------------------------------------
    # Learned base weights (normalized=std)
    # ------------------------------------------------------------
    w_lines: float = +0.689191
    w_holes: float = -0.297671
    w_max_height: float = -0.133701
    w_bumpiness: float = -0.121658
    w_agg_height: float = -0.174157

    # Learned bias (normalized=std)
    bias: float = +1.782983

    # ------------------------------------------------------------
    # Learned hinge weights (normalized=std)
    #   hinge(x; t) = max(0, x - t)
    # ------------------------------------------------------------

    # cleared_lines hinges
    w_cleared_lines_h0_t0: float = +0.689191  # max(0, cleared_lines - 0)

    # delta_holes hinges
    w_delta_holes_h0_t0: float = +0.027632  # max(0, delta_holes - 0)
    w_delta_holes_h1_t1: float = -0.054893  # max(0, delta_holes - 1)
    w_delta_holes_h2_t2: float = +0.043474  # max(0, delta_holes - 2)
    w_delta_holes_h3_t3: float = -0.011428  # max(0, delta_holes - 3)

    # delta_max_height hinges
    w_delta_max_height_h0_t0: float = +0.018693  # max(0, delta_max_height - 0)
    w_delta_max_height_h1_t1: float = +0.103864  # max(0, delta_max_height - 1)
    w_delta_max_height_h2_t2: float = +0.003546  # max(0, delta_max_height - 2)

    # delta_bumpiness hinges
    w_delta_bumpiness_h0_t0: float = +0.033758  # max(0, delta_bumpiness - 0)
    w_delta_bumpiness_h1_t1: float = -0.021236  # max(0, delta_bumpiness - 1)
    w_delta_bumpiness_h2_t2: float = +0.011389  # max(0, delta_bumpiness - 2)
    w_delta_bumpiness_h3_t4: float = +0.022555  # max(0, delta_bumpiness - 4)
    w_delta_bumpiness_h4_t6: float = -0.107050  # max(0, delta_bumpiness - 6)

    # delta_agg_height hinges
    w_delta_agg_height_h0_t4: float = -0.009259  # max(0, delta_agg_height - 4)
    w_delta_agg_height_h1_t5: float = +0.038531  # max(0, delta_agg_height - 5)
    w_delta_agg_height_h2_t6: float = -0.045231  # max(0, delta_agg_height - 6)
    w_delta_agg_height_h3_t7: float = +0.035550  # max(0, delta_agg_height - 7)

    # ------------------------------------------------------------
    # Penalties (POSITIVE magnitudes)
    # ------------------------------------------------------------
    illegal_penalty: float = 50.0
    terminal_penalty: float = 50.0

    @staticmethod
    def _hinge(x: float, t: float) -> float:
        return x - t if x > t else 0.0

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
        dmh = float(getattr(features, "delta_max_height", 0) or 0)
        db = float(getattr(features, "delta_bumpiness", 0) or 0)
        dah = float(getattr(features, "delta_agg_height", 0) or 0)

        # ------------------------------------------------------------
        # Base linear terms
        # ------------------------------------------------------------
        r += self.w_lines * cl
        r += self.w_holes * dh
        r += self.w_max_height * dmh
        r += self.w_bumpiness * db
        r += self.w_agg_height * dah

        # ------------------------------------------------------------
        # Piecewise hinge terms
        # ------------------------------------------------------------
        r += self.w_cleared_lines_h0_t0 * self._hinge(cl, 0.0)

        r += self.w_delta_holes_h0_t0 * self._hinge(dh, 0.0)
        r += self.w_delta_holes_h1_t1 * self._hinge(dh, 1.0)
        r += self.w_delta_holes_h2_t2 * self._hinge(dh, 2.0)
        r += self.w_delta_holes_h3_t3 * self._hinge(dh, 3.0)

        r += self.w_delta_max_height_h0_t0 * self._hinge(dmh, 0.0)
        r += self.w_delta_max_height_h1_t1 * self._hinge(dmh, 1.0)
        r += self.w_delta_max_height_h2_t2 * self._hinge(dmh, 2.0)

        r += self.w_delta_bumpiness_h0_t0 * self._hinge(db, 0.0)
        r += self.w_delta_bumpiness_h1_t1 * self._hinge(db, 1.0)
        r += self.w_delta_bumpiness_h2_t2 * self._hinge(db, 2.0)
        r += self.w_delta_bumpiness_h3_t4 * self._hinge(db, 4.0)
        r += self.w_delta_bumpiness_h4_t6 * self._hinge(db, 6.0)

        r += self.w_delta_agg_height_h0_t4 * self._hinge(dah, 4.0)
        r += self.w_delta_agg_height_h1_t5 * self._hinge(dah, 5.0)
        r += self.w_delta_agg_height_h2_t6 * self._hinge(dah, 6.0)
        r += self.w_delta_agg_height_h3_t7 * self._hinge(dah, 7.0)

        # bias
        r += self.bias

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
