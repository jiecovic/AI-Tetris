# src/tetris_rl/env_bundles/rewards/heuristic_softplus.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import math

from tetris_rl.envs.api import RewardFn, TransitionFeatures
from tetris_rl.envs.rewards.heuristics import DeltaHeuristicWeights, delta_heuristic_score


def _softplus(x: float) -> float:
    # numerically stable softplus
    # softplus(x) = log(1 + exp(x))
    if x > 50.0:
        return x
    if x < -50.0:
        return 0.0
    return float(math.log1p(math.exp(x)))


@dataclass(frozen=True)
class HeuristicSoftplusReward(RewardFn):
    """
    Reward = f(score) - clamp_penalty - terminal_penalty

    where:
      score = g(deltas, cleared_lines)
      f(score) = scale * softplus((score - shift) / tau)

    Properties:
      - f(score) >= 0 always (no suicide-by-negative-drift incentive for the shaped part)
      - smooth approach to 0 for bad moves
      - unbounded positive, but controlled by (tau, scale) and (optionally) reward_max

    Practical PPO note:
      Rare but extreme "score" outliers (e.g., weird overhang placements) can otherwise dominate
      policy updates. We therefore default to a modest (tau, scale) pair and enable reward_max
      to cap the shaped part.
    """

    weights: DeltaHeuristicWeights = DeltaHeuristicWeights()

    # Softplus shaping knobs
    # Heuristic starting point (tuned for typical 10x20 tetromino play):
    #   - shift ~= median(score) (0 is a reasonable initial guess)
    #   - tau sets the score scale where rewards become meaningfully > 0 (bigger = smoother)
    #   - scale sets overall magnitude; keep scale/tau modest to avoid huge rewards from rare outliers
    tau: float = 4.0
    shift: float = 0.0
    scale: float = 0.2

    clamp_penalty: float = 1.0
    game_over_penalty: float = 50.0

    # Optional clamp for PPO stability.
    # IMPORTANT: clamp applies to the shaped part ONLY, before penalties.
    reward_min: float | None = 0.0
    reward_max: float | None = 3.0

    def __call__(
            self,
            *,
            prev_state: Any,
            action: Any,
            next_state: Any,
            features: TransitionFeatures,
            info: Dict[str, Any],
    ) -> float:
        s = delta_heuristic_score(features=features, w=self.weights)

        tau = float(self.tau)
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")

        z = (float(s) - float(self.shift)) / tau
        r = float(self.scale) * _softplus(z)

        # --- clamp shaped reward ONLY (keep penalties effective) --------------------
        r_min = self.reward_min
        if r_min is not None and r < float(r_min):
            r = float(r_min)

        r_max = self.reward_max
        if r_max is not None and r > float(r_max):
            r = float(r_max)

        # --- penalties --------------------------------------------------------------
        if bool(features.clamped):
            r -= float(self.clamp_penalty)

        if bool(features.game_over):
            r -= float(self.game_over_penalty)

        return float(r)
