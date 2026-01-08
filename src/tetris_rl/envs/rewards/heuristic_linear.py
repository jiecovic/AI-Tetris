# src/tetris_rl/env_bundles/rewards/heuristic_linear.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
class HeuristicLinear(RewardFn):
    """
    Placement-level heuristic linear reward.

    Learned via linear regression from expert preference data
    (bc_state_heuristic_v6).

        r =
          +4.489054 * placed_cells_all_cleared
          -0.561582 * delta_holes
          -0.163864 * delta_max_height
          -0.125758 * delta_bumpiness
          +1.534284

    Conventions:
      - Weights already include sign.
      - Illegal actions receive only penalties (no learned terms).
      - Terminal penalty is applied if game_over is True.
    """

    # === Learned coefficients (SIGNED) ===
    w_placed_cells_all_cleared: float = +4.489054
    w_delta_holes: float = -0.561582
    w_delta_max_height: float = -0.163864
    w_delta_bumpiness: float = -0.125758

    # Learned bias (SIGNED)
    bias: float = +1.534284

    # Penalties (POSITIVE magnitudes)
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

        # deltas (missing -> 0.0)
        dh = float(getattr(features, "delta_holes", 0) or 0)
        dmh = float(getattr(features, "delta_max_height", 0) or 0)
        db = float(getattr(features, "delta_bumpiness", 0) or 0)

        # boolean feature (missing -> 0)
        p_all = 1.0 if bool(getattr(features, "placed_cells_all_cleared", False)) else 0.0

        # learned linear reward
        r += float(self.w_placed_cells_all_cleared) * p_all
        r += float(self.w_delta_holes) * dh
        r += float(self.w_delta_max_height) * dmh
        r += float(self.w_delta_bumpiness) * db
        r += float(self.bias)

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
