# src/tetris_rl/env_bundles/rewards/heuristic_phi_window.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass
class HeuristicPhiWindowReward(RewardFn):
    """
    Chunked / delayed reward using a linear heuristic potential Φ over a K-step window.

    Potential (maximize):
      Φ_shape(s) = a * agg_height(s) + c * holes(s) + d * bumpiness(s)

    Window reward:
      R = (Φ_shape(s_end) - Φ_shape(s_start))
          + b * sum_cleared_lines_in_window
          - invalid_action_penalty * (#invalid_actions_in_window)
          - terminal_penalty * [game_over flush]

    Reset detection:
      - Resets window state on env.reset(), detected via info["ui"]["episode_step"] == 0.
      - Flushes and resets on features.game_over.

    IMPORTANT:
      - Must be instantiated per-env instance (do not share one object across VecEnv workers).
      - Uses TransitionFeatures.{agg_height, holes, bumpiness} if present; None-safe fallbacks.
    """

    # Window length (steps per payout).
    k: int = 2

    # Heuristic weights (defaults match a common Tetris heuristic baseline).
    # a: aggregate height (negative)
    # b: cleared lines (positive) -- applied to summed cleared lines in the window
    # c: holes (negative)
    # d: bumpiness (negative)
    a: float = -0.510066
    b: float = 0.760666
    c: float = -0.35663
    d: float = -0.184483

    # Extra safety terms (optional).
    invalid_action_penalty: float = 10.0
    terminal_penalty: float = 50.0

    # --- internal state (per env instance) ---
    _t: int = field(default=0, init=False, repr=False)
    _sum_cleared: int = field(default=0, init=False, repr=False)
    _sum_invalid: int = field(default=0, init=False, repr=False)
    _phi_start: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.k = int(self.k)
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")
        if float(self.invalid_action_penalty) < 0.0:
            raise ValueError(f"invalid_action_penalty must be >= 0, got {self.invalid_action_penalty}")
        if float(self.terminal_penalty) < 0.0:
            raise ValueError(f"terminal_penalty must be >= 0, got {self.terminal_penalty}")

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _reset_window(self) -> None:
        self._t = 0
        self._sum_cleared = 0
        self._sum_invalid = 0
        self._phi_start = None

    def _episode_step(self, info: Dict[str, Any]) -> Optional[int]:
        ui = info.get("ui")
        if isinstance(ui, dict):
            v = ui.get("episode_step")
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None
        return None

    def _phi_shape(self, features: TransitionFeatures) -> float:
        # None-safe: treat missing metrics as 0 (neutral).
        agg_h = getattr(features, "agg_height", None)
        holes = getattr(features, "holes", None)
        bump = getattr(features, "bumpiness", None)

        ah = float(int(agg_h)) if agg_h is not None else 0.0
        ho = float(int(holes)) if holes is not None else 0.0
        bu = float(int(bump)) if bump is not None else 0.0

        return float(self.a) * ah + float(self.c) * ho + float(self.d) * bu

    # ---------------------------------------------------------------------
    # RewardFn
    # ---------------------------------------------------------------------

    def __call__(
        self,
        *,
        prev_state: Any,
        action: Any,
        next_state: Any,
        features: TransitionFeatures,
        info: Dict[str, Any],
    ) -> float:
        # Reset window on new episode boundary (env.reset()).
        ep_step = self._episode_step(info)
        if ep_step == 0:
            self._reset_window()
            return 0.0

        # Start-of-window potential snapshot.
        if self._t == 0:
            self._phi_start = float(self._phi_shape(features))

        # Accumulate step stats.
        self._t += 1

        cl = int(getattr(features, "cleared_lines", 0) or 0)
        if cl < 0:
            cl = 0
        self._sum_cleared += cl

        if bool(getattr(features, "invalid_action", False)):
            self._sum_invalid += 1

        # Window ends at k steps, or flush early on game over.
        flush = bool(getattr(features, "game_over", False))
        chunk_end = bool(self._t >= int(self.k))

        if not (flush or chunk_end):
            return 0.0

        # --- payout ---
        phi_end = float(self._phi_shape(features))
        phi_start = float(self._phi_start) if self._phi_start is not None else 0.0

        r = 0.0
        r += (phi_end - phi_start)
        r += float(self.b) * float(self._sum_cleared)

        if float(self.invalid_action_penalty) > 0.0 and self._sum_invalid > 0:
            r -= float(self.invalid_action_penalty) * float(self._sum_invalid)

        if flush and float(self.terminal_penalty) > 0.0:
            r -= float(self.terminal_penalty)

        # Reset for next window.
        self._reset_window()
        return float(r)


__all__ = ["HeuristicPhiWindowReward"]
