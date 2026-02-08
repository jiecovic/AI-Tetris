# src/tetris_rl/core/policies/planning_policies/heuristic_policy.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from planning_rl.policies import VectorParamPolicy
from tetris_rl.core.policies.spec import (
    HeuristicSearch,
    HeuristicSpec,
    load_heuristic_spec,
    save_heuristic_spec,
)


class HeuristicPlanningPolicy(VectorParamPolicy):
    def __init__(
        self,
        *,
        features: Sequence[str],
        search: HeuristicSearch | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.features = list(features)
        self.search = search or HeuristicSearch()
        self._spec: HeuristicSpec | None = None
        self._policy: Any | None = None
        if not self.features:
            raise ValueError("features must be non-empty")
        if weights is not None:
            self.set_params(weights)

    @property
    def num_params(self) -> int:
        return len(self.features)

    def get_params(self) -> Sequence[float]:
        if self._spec is None:
            raise RuntimeError("policy params not set; call set_params() first")
        return list(self._spec.weights)

    def build_spec(self, weights: Sequence[float]) -> HeuristicSpec:
        if len(weights) != len(self.features):
            raise ValueError("weights length must match features length")
        return HeuristicSpec(
            features=list(self.features),
            weights=list(weights),
            search=self.search,
        )

    def _build_policy(self, spec: HeuristicSpec) -> Any:
        from tetris_rl_engine import ExpertPolicy

        search = spec.search
        clear_mode = str(search.feature_clear_mode).strip().lower()
        after_clear = clear_mode in {"post", "clear"}
        return ExpertPolicy.heuristic(
            features=list(spec.features),
            weights=list(spec.weights),
            plies=int(search.plies),
            beam_width=None if search.beam_width is None else int(search.beam_width),
            beam_from_depth=int(search.beam_from_depth),
            after_clear=bool(after_clear),
        )

    def set_params(self, params: Sequence[float]) -> None:
        self._spec = self.build_spec(params)
        self._policy = self._build_policy(self._spec)

    def state_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "features": list(self.features),
            "search": self.search.model_dump(mode="json"),
        }
        if self._spec is not None:
            data["params"] = list(self._spec.weights)
        return data

    def load_state_dict(self, state: dict[str, Any]) -> None:
        features = state.get("features")
        if isinstance(features, list) and features:
            self.features = list(features)
        search = state.get("search")
        if isinstance(search, dict):
            self.search = HeuristicSearch.model_validate(search)
        params = state.get("params")
        if params is not None:
            self.set_params(params)

    @classmethod
    def from_spec(cls, spec: HeuristicSpec) -> "HeuristicPlanningPolicy":
        return cls(features=spec.features, search=spec.search, weights=spec.weights)

    @classmethod
    def from_yaml(cls, path: Path) -> "HeuristicPlanningPolicy":
        return cls.from_spec(load_heuristic_spec(path))

    def save_spec(self, path: Path) -> Path:
        if self._spec is None:
            raise RuntimeError("policy params not set; call set_params() first")
        return save_heuristic_spec(path, self._spec)

    def action_id(self, game: Any) -> int | None:
        if self._policy is None:
            raise RuntimeError("policy params not set; call set_params() first")
        return self._policy.action_id(game)

    def action_for_env(self, *, env: Any, game: Any) -> Any:
        aid = self.action_id(game)
        if aid is None:
            aid = 0
        action_mode = str(getattr(env, "action_mode", "discrete")).strip().lower()
        if action_mode == "discrete":
            return int(aid)
        rot_u, col_u = game.decode_action_id(int(aid))
        return (int(rot_u), int(col_u))

    def predict(self, *, env: Any) -> Any:
        if self._policy is None:
            raise RuntimeError("policy params not set; call set_params() first")
        game = getattr(env, "game", None)
        if game is None:
            raise ValueError("env must expose .game for heuristic predict")
        return self.action_for_env(env=env, game=game)


__all__ = ["HeuristicPlanningPolicy"]
