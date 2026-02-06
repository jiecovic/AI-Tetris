# src/tetris_rl/env_bundles/rewards/heuristic_expert_shaping.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tetris_rl.envs.api import RewardFn, TransitionFeatures


@dataclass
class HeuristicExpertShaping(RewardFn):
    """
    Combined reward shaping:

      (A) Learned ridge delta reward (linear model on placement deltas)
      (B) Weak expert match shaping (default: codemy0())
      (C) Small Tetris bonus (4-line clear)
      (D) Safety penalties

    Notes:
      - If invalid_action is True, we apply penalties + (optional) expert shaping + game_over penalty,
        but we skip the learned linear terms (mirrors the original LearnedRidgeDeltaReward behavior).
      - Expert shaping expects env.step() to inject a transient engine ref into:
            info["engine_info"]["game"]
        (Do NOT return the PyO3 object in the final info dict passed to SB3 VecEnv; inject only for
        reward computation to avoid deepcopy/pickle errors.)
      - expert has: action_id(game) -> Optional[int]
    """

    # ------------------------------------------------------------------
    # Expert shaping (optional)
    # ------------------------------------------------------------------
    expert_policy: Optional[Any] = None

    # keep small!
    expert_match_bonus: float = 0.5
    expert_mismatch_penalty: float = 0.0

    # ------------------------------------------------------------------
    # Learned ridge delta model (SIGNED coefficients)
    # ------------------------------------------------------------------
    w_cleared_lines: float = +1.774651
    w_delta_holes: float = -0.477397
    w_delta_max_height: float = -0.067596
    w_delta_bumpiness: float = -0.106060

    # learned bias (set to 0.0 if you want "pure shaping" without constant offset)
    bias: float = 0.01

    # ------------------------------------------------------------------
    # Tetris-specific bonus
    # ------------------------------------------------------------------
    tetris_bonus: float = 4.0

    well_opening_bonus: float = 1.0

    # ------------------------------------------------------------------
    # Safety penalties (POSITIVE magnitudes)
    # ------------------------------------------------------------------
    invalid_action_penalty: float = 50.0
    game_over_penalty: float = 100.0

    _expert_cached: Optional[Any] = field(default=None, init=False, repr=False)

    def _get_expert(self) -> Any:
        if self._expert_cached is not None:
            return self._expert_cached

        if self.expert_policy is not None:
            self._expert_cached = self.expert_policy
            return self._expert_cached

        from tetris_rl_engine import ExpertPolicy

        self._expert_cached = ExpertPolicy.codemy0()
        return self._expert_cached

    def _expert_shaping(self, *, features: TransitionFeatures, info: Dict[str, Any]) -> float:
        engine_info = info.get("engine_info", {})
        game = engine_info.get("game", None)

        used_aid_raw = getattr(features, "used_action_id", None)
        used_aid: Optional[int] = used_aid_raw if isinstance(used_aid_raw, int) else None

        if game is None or used_aid is None:
            return 0.0

        expert = self._get_expert()
        try:
            expert_aid = expert.action_id(game)
        except Exception:
            expert_aid = None

        if expert_aid is None:
            return 0.0

        if int(used_aid) == int(expert_aid):
            return float(self.expert_match_bonus)
        return -float(self.expert_mismatch_penalty)

    def _learned_ridge_terms(self, *, features: TransitionFeatures) -> float:
        # placed cells cleared (missing -> 0)
        pcc = int(getattr(features, "cleared_lines", 0) or 0)

        # deltas (missing -> 0.0)
        dh = float(getattr(features, "delta_holes", 0) or 0)
        dmh = float(getattr(features, "delta_max_height", 0) or 0)
        db = float(getattr(features, "delta_bumpiness", 0) or 0)

        r = 0.0
        r += float(self.w_cleared_lines) * float(pcc)
        r += float(self.w_delta_holes) * float(dh)
        r += float(self.w_delta_max_height) * float(dmh)
        r += float(self.w_delta_bumpiness) * float(db)
        r += float(self.bias)
        return float(r)

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

        cleared = int(getattr(features, "cleared_lines", 0) or 0)
        delta_holes = int(getattr(features, "delta_holes", 0) or 0)
        invalid = bool(getattr(features, "invalid_action", False))
        game_over = bool(getattr(features, "game_over", False))

        if delta_holes < 0:
            r += float(self.well_opening_bonus)

        # ------------------------------------------------------------
        # (1) Learned ridge terms (skip on invalid actions)
        # ------------------------------------------------------------
        if not invalid:
            r += float(self._learned_ridge_terms(features=features))

        # ------------------------------------------------------------
        # (2) Tetris bonus
        # ------------------------------------------------------------
        if cleared == 4:
            r += float(self.tetris_bonus)

        # ------------------------------------------------------------
        # (3) Expert shaping (applies if engine is available)
        # ------------------------------------------------------------
        r += float(self._expert_shaping(features=features, info=info))

        # ------------------------------------------------------------
        # (4) Safety penalties
        # ------------------------------------------------------------
        if invalid:
            # keep your original heuristic penalty
            r -= float(self.invalid_action_penalty)
            # optional stronger penalty (if enabled)

        if game_over:
            r -= float(self.game_over_penalty)
            # optional stronger penalty (if enabled)

        return float(r)
