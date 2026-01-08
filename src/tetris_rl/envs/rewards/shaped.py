# src/tetris_rl/env_bundles/rewards/shaped.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
class ShapedMacroReward(RewardFn):
    """
    Shaped reward for the macro (placement) environment.

    Conventions:
      - All penalties are POSITIVE magnitudes.
      - Penalties are SUBTRACTED at computation time.

    Illegal actions:
      - Subtract illegal_penalty.
      - Receive NO survival reward.
      - Terminal penalty is still subtracted if game_over is True.
    """

    # Base per-step survival reward
    survival_r: float = 0.1

    # Hole penalty expressed as a reduction of survival (negative per new hole)
    hole_r: float = -0.05  # per newly created hole

    # Line clears
    line_r: float = 1.0
    line_alpha: float = 1.0

    # Penalties (POSITIVE magnitudes)
    illegal_penalty: float = 0.25
    terminal_penalty: float = 2.0

    def __call__(
        self,
        *,
        prev_state: Any,
        action: Any,
        next_state: Any,
        features: TransitionFeatures,
        info: Dict[str, Any],
    ) -> float:
        r_total = 0.0

        # ------------------------------------------------------------
        # Illegal action: penalties only
        # ------------------------------------------------------------
        if bool(getattr(features, "illegal_action", False)):
            r_total -= float(self.illegal_penalty)
            if bool(getattr(features, "game_over", False)):
                r_total -= float(self.terminal_penalty)
            return float(r_total)

        # ------------------------------------------------------------
        # Hole adjustment (only new holes)
        # ------------------------------------------------------------
        delta_holes = getattr(features, "delta_holes", None)
        new_holes = max(0, int(delta_holes)) if delta_holes is not None else 0

        r_holes = 0.0
        if new_holes > 0:
            r_holes = self.hole_r * new_holes  # hole_r already negative

        # ------------------------------------------------------------
        # Line clears
        # ------------------------------------------------------------
        r_lines = 0.0
        cleared = int(getattr(features, "cleared_lines", 0) or 0)
        if cleared > 0:
            r_lines = self.line_r * (cleared ** self.line_alpha)

        # ------------------------------------------------------------
        # Survival reward
        # ------------------------------------------------------------
        r_survival_eff = max(0.0, self.survival_r + r_holes + r_lines)
        r_total += r_survival_eff

        # ------------------------------------------------------------
        # Terminal penalty
        # ------------------------------------------------------------
        if bool(getattr(features, "game_over", False)):
            r_total -= float(self.terminal_penalty)

        return float(r_total)
