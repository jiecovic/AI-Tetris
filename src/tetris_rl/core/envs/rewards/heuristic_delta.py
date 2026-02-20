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

    Optional clipping:
      - clip_term_lines / clip_term_holes / clip_term_bumpiness / clip_term_agg_height
        clamp each weighted contribution term to [-clip, +clip]

    Optional survival scaling:
      - survival_scale_mode: none|max_h|agg_h
      - survival_scale_floor: minimum scale in [0,1]
      - survival_scale_power: >=1, sharpness of high-board penalty

    Extra bonus:
      - tetris_bonus is added when exactly 4 lines are cleared.

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
        self.tetris_bonus = float(spec.tetris_bonus)
        self.clip_term_lines = float(spec.clip_term_lines) if spec.clip_term_lines is not None else None
        self.clip_term_holes = float(spec.clip_term_holes) if spec.clip_term_holes is not None else None
        self.clip_term_bumpiness = float(spec.clip_term_bumpiness) if spec.clip_term_bumpiness is not None else None
        self.clip_term_agg_height = float(spec.clip_term_agg_height) if spec.clip_term_agg_height is not None else None
        self.survival_bonus = float(spec.survival_bonus)
        self.survival_scale_mode = str(spec.survival_scale_mode).strip().lower()
        self.survival_scale_floor = float(spec.survival_scale_floor)
        self.survival_scale_power = float(spec.survival_scale_power)

    @staticmethod
    def _clip_symmetric(value: float, clip: float | None) -> float:
        if clip is None:
            return float(value)
        c = float(clip)
        if c <= 0.0:
            return 0.0
        return float(max(-c, min(c, float(value))))

    @staticmethod
    def _safe_int(x: Any) -> int | None:
        try:
            if x is None:
                return None
            return int(x)
        except Exception:
            return None

    def _resolve_board_hw(self, *, next_state: Any, info: Dict[str, Any]) -> tuple[int | None, int | None]:
        try:
            engine_info = info.get("engine_info", None)
            if isinstance(engine_info, dict):
                game = engine_info.get("game", None)
                if game is not None:
                    h = self._safe_int(getattr(game, "visible_h", lambda: None)())
                    w = self._safe_int(getattr(game, "board_w", lambda: None)())
                    if h is not None and w is not None and h > 0 and w > 0:
                        return h, w
        except Exception:
            pass

        if isinstance(next_state, dict):
            grid = next_state.get("grid", None)
            if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                h = self._safe_int(len(grid))
                w = self._safe_int(len(grid[0]))
                if h is not None and w is not None and h > 0 and w > 0:
                    return h, w
        return None, None

    def _survival_scale(
        self,
        *,
        features: TransitionFeatures,
        next_state: Any,
        info: Dict[str, Any],
    ) -> float:
        mode = str(self.survival_scale_mode).strip().lower()
        if mode == "none":
            return 1.0

        h, w = self._resolve_board_hw(next_state=next_state, info=info)
        if h is None or h <= 0:
            return 1.0

        if mode == "max_h":
            max_h = self._safe_int(getattr(features, "max_height", None))
            if max_h is None:
                return 1.0
            risk = float(max_h) / float(h)
        elif mode == "agg_h":
            if w is None or w <= 0:
                return 1.0
            agg_h = self._safe_int(getattr(features, "agg_height", None))
            if agg_h is None:
                return 1.0
            risk = float(agg_h) / float(h * w)
        else:
            return 1.0

        risk = max(0.0, min(1.0, float(risk)))
        base = max(0.0, 1.0 - risk)
        shaped = float(base) ** float(self.survival_scale_power)
        floor = max(0.0, min(1.0, float(self.survival_scale_floor)))
        return float(floor + (1.0 - floor) * shaped)

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

        line_term = float(self.w_lines) * float(cl)
        holes_term = float(self.w_holes) * float(dh)
        bump_term = float(self.w_bumpiness) * float(db)
        agg_term = float(self.w_agg_height) * float(dah)

        line_term = self._clip_symmetric(line_term, self.clip_term_lines)
        holes_term = self._clip_symmetric(holes_term, self.clip_term_holes)
        bump_term = self._clip_symmetric(bump_term, self.clip_term_bumpiness)
        agg_term = self._clip_symmetric(agg_term, self.clip_term_agg_height)

        # CodemyRoad delta-shaped reward (post-weight term clipping)
        r += line_term
        if cl == 4:
            r += float(self.tetris_bonus)
        r += holes_term
        r += bump_term
        r += agg_term

        # Optional per-step bias, optionally scaled by board risk.
        r += float(self.survival_bonus) * self._survival_scale(
            features=features,
            next_state=next_state,
            info=info,
        )

        # terminal penalty
        if bool(getattr(features, "game_over", False)):
            r -= float(self.terminal_penalty)

        return float(r)
