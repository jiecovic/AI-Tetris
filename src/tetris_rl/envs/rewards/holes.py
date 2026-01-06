# src/tetris_rl/envs/rewards/holes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass(frozen=True)
class HolesDeltaReward(RewardFn):
    """
    Placement-level reward shaping using only:
      - line clear reward, and
      - delta holes (punish new holes; optionally weak credit for improvements),
    plus:
      - clamp penalty
      - terminal penalty on game over

    Reward:
      r = line_rewards[cleared_lines]
          - w_new_holes * pos(delta_holes)
          + (w_new_holes * improve_frac) * pos(-delta_holes)
          - clamp_penalty * [clamped]
          - game_over_penalty * [game_over]

    Notes:
      - pos(x) = max(0, x)
      - Improvements are deliberately weaker (improve_frac) to reduce oscillation incentives.
      - cleared_lines is hard-capped at 4 (never error): treat any >=4 as "at least a Tetris".
    """

    # Index = cleared_lines in {0..4}.
    line_rewards: tuple[float, float, float, float, float] = (0.0, 1.0, 3.0, 5.0, 8.0)

    # Penalize *increases* in holes.
    w_new_holes: float = 1.5

    # Optional: small credit for *reducing* holes (negative delta).
    # Keep this < 1.0 to ensure penalties dominate.
    improve_frac: float = 0.25

    # Clamp penalty when action got forced into a legal placement.
    clamp_penalty: float = 1.0

    # Terminal penalty on game over.
    game_over_penalty: float = 50.0

    def __call__(
            self,
            *,
            prev_state: Any,
            action: Any,
            next_state: Any,
            features: TransitionFeatures,
            info: Dict[str, Any],
    ) -> float:
        # --- (1) Line clear reward ---------------------------------------------------
        cl = int(getattr(features, "cleared_lines", 0) or 0)
        cl = max(0, min(cl, 4))  # hard cap: never error out (supports alt rules / bug tolerance)
        r = float(self.line_rewards[cl])

        # --- (2) Delta holes shaping -------------------------------------------------
        dh = float(getattr(features, "delta_holes", 0.0) or 0.0)

        # Punish new holes (positive delta).
        if dh > 0.0:
            r -= float(self.w_new_holes) * dh

        # Optional weak credit for improvements (negative delta).
        if dh < 0.0 and float(self.improve_frac) > 0.0:
            r += float(self.w_new_holes) * float(self.improve_frac) * (-dh)

        # --- (3) Clamp ---------------------------------------------------------------
        if bool(getattr(features, "clamped", False)):
            r -= float(self.clamp_penalty)

        # --- (4) Terminal ------------------------------------------------------------
        if bool(getattr(features, "game_over", False)):
            r -= float(self.game_over_penalty)

        return float(r)
