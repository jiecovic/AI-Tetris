# src/tetris_rl/env_bundles/rewards/lines.py
from __future__ import annotations

from typing import Any, Dict

from tetris_rl.envs.api import RewardFn, TransitionFeatures


class LinesReward(RewardFn):
    """
    Baseline reward: number of lines cleared on the placement.
    """

    def __call__(
            self,
            *,
            prev_state: Any,
            action: Any,
            next_state: Any,
            features: TransitionFeatures,
            info: Dict[str, Any],
    ) -> float:
        return float(features.cleared_lines)
