# src/tetris_rl/core/envs/rewards/lines_shape.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.core.envs.api import RewardFn, TransitionFeatures
from tetris_rl.core.envs.rewards.params import LinesShapeRewardParams


class LinesShapeReward(RewardFn):
    """
    Reward shaping from board deltas and special line clears.

    Rules (params control magnitudes):
      - +line_cleared_bonus * cleared_lines (per-line scaling).
      - +<metric>_increase_reward if delta_metric > 0.
      - +<metric>_same_reward if delta_metric == 0.
      - +<metric>_decrease_reward if delta_metric < 0.
        metrics: holes, bumpiness, max_height, agg_height
      - +tetris_bonus if cleared_lines == 4.
    """

    def __init__(self, *, spec: LinesShapeRewardParams) -> None:
        self.invalid_penalty = float(spec.invalid_penalty)
        self.terminal_penalty = float(spec.terminal_penalty)
        self.survival_bonus = float(spec.survival_bonus)
        self.line_cleared_bonus = float(spec.line_cleared_bonus)
        self.holes_increase_reward = float(spec.holes_increase_reward)
        self.holes_same_reward = float(spec.holes_same_reward)
        self.holes_decrease_reward = float(spec.holes_decrease_reward)
        self.bumpiness_increase_reward = float(spec.bumpiness_increase_reward)
        self.bumpiness_same_reward = float(spec.bumpiness_same_reward)
        self.bumpiness_decrease_reward = float(spec.bumpiness_decrease_reward)
        self.max_height_increase_reward = float(spec.max_height_increase_reward)
        self.max_height_same_reward = float(spec.max_height_same_reward)
        self.max_height_decrease_reward = float(spec.max_height_decrease_reward)
        self.agg_height_increase_reward = float(spec.agg_height_increase_reward)
        self.agg_height_same_reward = float(spec.agg_height_same_reward)
        self.agg_height_decrease_reward = float(spec.agg_height_decrease_reward)
        self.tetris_bonus = float(spec.tetris_bonus)

    @staticmethod
    def _delta_triplet_reward(
        *,
        delta: float,
        increase_reward: float,
        same_reward: float,
        decrease_reward: float,
    ) -> float:
        if float(delta) > 0.0:
            return float(increase_reward)
        if float(delta) < 0.0:
            return float(decrease_reward)
        return float(same_reward)

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

        if invalid:
            r -= float(self.invalid_penalty)
            if game_over:
                r -= float(self.terminal_penalty)
            return float(r)

        d_holes = float(getattr(features, "delta_holes", 0) or 0)
        d_bumpiness = float(getattr(features, "delta_bumpiness", 0) or 0)
        d_max_height = float(getattr(features, "delta_max_height", 0) or 0)
        d_agg_height = float(getattr(features, "delta_agg_height", 0) or 0)

        cleared = int(getattr(features, "cleared_lines", 0) or 0)
        cleared = max(0, min(cleared, 4))

        if cleared > 0:
            r += float(self.line_cleared_bonus) * float(cleared)

        r += self._delta_triplet_reward(
            delta=d_holes,
            increase_reward=self.holes_increase_reward,
            same_reward=self.holes_same_reward,
            decrease_reward=self.holes_decrease_reward,
        )
        r += self._delta_triplet_reward(
            delta=d_bumpiness,
            increase_reward=self.bumpiness_increase_reward,
            same_reward=self.bumpiness_same_reward,
            decrease_reward=self.bumpiness_decrease_reward,
        )
        r += self._delta_triplet_reward(
            delta=d_max_height,
            increase_reward=self.max_height_increase_reward,
            same_reward=self.max_height_same_reward,
            decrease_reward=self.max_height_decrease_reward,
        )
        r += self._delta_triplet_reward(
            delta=d_agg_height,
            increase_reward=self.agg_height_increase_reward,
            same_reward=self.agg_height_same_reward,
            decrease_reward=self.agg_height_decrease_reward,
        )

        if cleared == 4:
            r += float(self.tetris_bonus)

        if not game_over:
            r += float(self.survival_bonus)

        if game_over:
            r -= float(self.terminal_penalty)

        return float(r)


__all__ = ["LinesShapeReward"]
