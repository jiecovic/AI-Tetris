# src/tetris_rl/env_bundles/rewards/learned_ridge_delta.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
class LearnedRidgeDeltaReward(RewardFn):
    """
    Placement-level reward shaping learned via ridge regression
    from expert preference data.

    Direct transcription of the fitted linear model (post-normalization):

        r =
          +1.774651 * placed_cells_cleared
          -0.477397 * delta_holes
          -0.067596 * delta_max_height
          -0.106060 * delta_bumpiness
          +1.160576

    Conventions:
      - Learned weights already include sign.
      - Illegal actions receive only penalties (no learned terms).
      - Terminal penalty is still applied if game_over is True.
    """

    # === Learned coefficients (SIGNED) ===
    w_placed_cells_cleared: float = +1.774651
    w_delta_holes: float = -0.477397
    w_delta_max_height: float = -0.067596
    w_delta_bumpiness: float = -0.106060

    # Learned bias (SIGNED)
    # bias: float = +1.160576
    bias: float = 0.5

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
        if bool(getattr(features, "invalid_action", False)):
            r -= float(self.illegal_penalty)
            if bool(getattr(features, "game_over", False)):
                r -= float(self.terminal_penalty)
            return float(r)

        # placed cells cleared (missing -> 0)
        pcc = int(getattr(features, "placed_cells_cleared", 0) or 0)

        # deltas (missing -> 0.0)
        dh = float(getattr(features, "delta_holes", 0) or 0)
        dmh = float(getattr(features, "delta_max_height", 0) or 0)
        db = float(getattr(features, "delta_bumpiness", 0) or 0)

        # learned linear reward (all signs live in weights)
        r += float(self.w_placed_cells_cleared) * float(pcc)
        r += float(self.w_delta_holes) * float(dh)
        r += float(self.w_delta_max_height) * float(dmh)
        r += float(self.w_delta_bumpiness) * float(db)
        r += float(self.bias)

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
