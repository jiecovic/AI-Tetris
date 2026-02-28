# src/tetris_rl/core/envs/rewards/lines_height_scaled.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import LinesHeightScaledRewardParams


class LinesHeightScaledReward(RewardFn):
    """
    Lines-style reward with height-aware scaling on positive terms only.

    High-level behavior:
      - Keep the same positive signal family as `lines`:
          line reward + optional tetris bonus + optional survival bonus.
      - Reduce those positive rewards as the board fills up, using normalized
        aggregate height risk.
      - Keep penalties unscaled so invalid/game-over consequences stay strong.

    Computation:
      1) Build positive component (for valid actions):
           positive = line_cleared_bonus * clamp(cleared_lines, 0..4)
                     + (tetris_bonus if cleared_lines == 4 else 0)
                     + (board_clear_bonus if board is empty else 0)
                     + (survival_bonus if not game_over else 0)

      2) Compute risk from board occupancy:
           if positive_scale_metric == "agg_height":
             risk = agg_height / (board_h * board_w)
           else:
             risk = max_height / board_h
           (clamped to [0, 1])

      3) Convert risk to scale:
            scale = floor + (1 - floor) * (1 - risk) ** power

          where:
            floor = positive_scale_floor in [0,1]
            power = positive_scale_power >= 1
            positive_scale_metric = agg_height | max_height

      4) Apply scaled positive reward:
           reward += positive * scale

    5) Apply truncation-only board penalty (when `truncated=True`):
            - height_term = -height_coeff * min(height_metric, height_cap)
            - holes_term  = -holes_coeff * min(holes, holes_cap)
            - terminal board metric can be `max_height` or `agg_height`

    6) Apply penalties (unscaled):
           - invalid action: reward -= invalid_penalty
           - terminal step:  reward -= terminal_penalty

    Fallbacks:
      - If selected metric or board dimensions cannot be resolved, scaling defaults
      to 1.0 (no attenuation) instead of failing.
    """

    def __init__(self, *, spec: LinesHeightScaledRewardParams) -> None:
        self.invalid_penalty = float(spec.invalid_penalty)
        self.terminal_penalty = float(spec.terminal_penalty)
        self.survival_bonus = float(spec.survival_bonus)
        self.line_cleared_bonus = float(spec.line_cleared_bonus)
        self.tetris_bonus = float(spec.tetris_bonus)
        self.board_clear_bonus = float(spec.board_clear_bonus)
        self.positive_scale_floor = float(spec.positive_scale_floor)
        self.positive_scale_power = float(spec.positive_scale_power)
        self.positive_scale_metric = str(spec.positive_scale_metric).strip().lower()
        self.truncation_penalty_height_coeff = float(spec.truncation_penalty_height_coeff)
        self.truncation_penalty_holes_coeff = float(spec.truncation_penalty_holes_coeff)
        self.truncation_height_metric = str(spec.truncation_height_metric).strip().lower()
        self.truncation_height_cap = float(spec.truncation_height_cap)
        self.truncation_holes_cap = float(spec.truncation_holes_cap)

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
            shape = getattr(grid, "shape", None)
            if shape is not None and len(shape) == 2:
                h = self._safe_int(shape[0])
                w = self._safe_int(shape[1])
                if h is not None and w is not None and h > 0 and w > 0:
                    return h, w
            if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                h = self._safe_int(len(grid))
                w = self._safe_int(len(grid[0]))
                if h is not None and w is not None and h > 0 and w > 0:
                    return h, w

        return None, None

    def _positive_scale(
        self,
        *,
        features: TransitionFeatures,
        next_state: Any,
        info: Dict[str, Any],
    ) -> float:
        h, w = self._resolve_board_hw(next_state=next_state, info=info)
        if h is None or h <= 0:
            return 1.0
        metric_name = self.positive_scale_metric
        if metric_name == "max_height":
            max_h = self._safe_int(getattr(features, "max_height", None))
            if max_h is None:
                return 1.0
            risk = float(max_h) / float(h)
        else:
            agg_h = self._safe_int(getattr(features, "agg_height", None))
            if agg_h is None or w is None or w <= 0:
                return 1.0
            risk = float(agg_h) / float(h * w)
        risk = max(0.0, min(1.0, float(risk)))
        base = max(0.0, 1.0 - risk)

        floor = max(0.0, min(1.0, float(self.positive_scale_floor)))
        shaped = float(base) ** float(self.positive_scale_power)
        return float(floor + (1.0 - floor) * shaped)

    def _truncation_penalty(self, *, features: TransitionFeatures) -> float:
        if self.truncation_penalty_height_coeff <= 0.0 and self.truncation_penalty_holes_coeff <= 0.0:
            return 0.0

        holes = self._safe_int(getattr(features, "holes", None))
        metric_name = self.truncation_height_metric
        if metric_name == "agg_height":
            raw_h = self._safe_int(getattr(features, "agg_height", None))
        else:
            raw_h = self._safe_int(getattr(features, "max_height", None))

        h = float(raw_h) if raw_h is not None else 0.0
        holes_f = float(holes) if holes is not None else 0.0

        h = min(h, float(self.truncation_height_cap))
        holes_f = min(holes_f, float(self.truncation_holes_cap))

        return float(-self.truncation_penalty_height_coeff * h - self.truncation_penalty_holes_coeff * holes_f)

    def _is_board_clear(self, *, features: TransitionFeatures, next_state: Any) -> bool:
        agg_h = self._safe_int(getattr(features, "agg_height", None))
        if agg_h is not None:
            return int(agg_h) <= 0

        max_h = self._safe_int(getattr(features, "max_height", None))
        if max_h is not None:
            return int(max_h) <= 0

        if isinstance(next_state, dict):
            grid = next_state.get("grid", None)
            try:
                any_fn = getattr(grid, "any", None)
                if callable(any_fn):
                    return not bool(any_fn())
            except Exception:
                pass
            if isinstance(grid, list):
                for row in grid:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        try:
                            if int(cell) != 0:
                                return False
                        except Exception:
                            if bool(cell):
                                return False
                return True
        return False

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
        invalid = bool(getattr(features, "invalid_action", False))
        game_over = bool(getattr(features, "game_over", False))
        truncated = bool(info.get("truncated", False))

        if invalid:
            r -= float(self.invalid_penalty)
            if game_over:
                r -= float(self.terminal_penalty)
            if truncated:
                r += float(self._truncation_penalty(features=features))
            return float(r)

        cleared = int(getattr(features, "cleared_lines", 0) or 0)
        cleared = max(0, min(cleared, 4))

        positive = float(self.line_cleared_bonus) * float(cleared)
        if cleared == 4:
            positive += float(self.tetris_bonus)
        if self._is_board_clear(features=features, next_state=next_state):
            positive += float(self.board_clear_bonus)
        if not game_over:
            positive += float(self.survival_bonus)

        r += float(positive) * self._positive_scale(
            features=features,
            next_state=next_state,
            info=info,
        )

        if game_over:
            r -= float(self.terminal_penalty)
        if truncated:
            r += float(self._truncation_penalty(features=features))

        return float(r)


__all__ = ["LinesHeightScaledReward"]
