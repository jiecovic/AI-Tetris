# src/tetris_rl/core/policies/planning_policies/td_value_policy.py
from __future__ import annotations

from typing import Any, Sequence

from planning_rl.td.policy import TDPolicy
from planning_rl.td.model import LinearValueModel
from tetris_rl.core.policies.planning_policies.heuristic_policy import HeuristicPlanningPolicy
from tetris_rl.core.policies.spec import HeuristicSearch


class TDValuePlanningPolicy(TDPolicy):
    def __init__(
        self,
        *,
        features: Sequence[str],
        search: HeuristicSearch | None,
        value_model: Any,
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise ValueError("features must be non-empty")
        self.search = search or HeuristicSearch()
        self._value_model = value_model
        self._heuristic = HeuristicPlanningPolicy(features=list(self.features), search=self.search)
        self._last_weights: list[float] | None = None
        self.sync_from_model()

    @property
    def value_model(self) -> Any:
        return self._value_model

    def _model_weights(self) -> list[float]:
        if hasattr(self._value_model, "get_weights"):
            weights = self._value_model.get_weights()
            return [float(w) for w in weights]
        state = getattr(self._value_model, "state_dict", lambda: {})()
        if isinstance(state, dict) and "weights" in state:
            w = state["weights"]
            try:
                return [float(x) for x in w.detach().cpu().tolist()]
            except Exception:
                return [float(x) for x in w]
        raise RuntimeError("value_model does not expose get_weights or weights state")

    def sync_from_model(self) -> None:
        weights = self._model_weights()
        if self._last_weights is not None and weights == self._last_weights:
            return
        self._heuristic.set_params(weights)
        self._last_weights = list(weights)

    def predict(self, *, env: Any) -> Any:
        if self._last_weights is None:
            self.sync_from_model()
        return self._heuristic.predict(env=env)

    def get_params(self) -> Sequence[float]:
        self.sync_from_model()
        return self._heuristic.get_params()

    def build_spec(self, weights: Sequence[float]) -> Any:
        return self._heuristic.build_spec(weights)

    def state_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "search": self.search.model_dump(mode="json"),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        features = state.get("features")
        if isinstance(features, list) and features:
            self.features = list(features)
        search = state.get("search")
        if isinstance(search, dict):
            self.search = HeuristicSearch.model_validate(search)
        self._heuristic = HeuristicPlanningPolicy(features=list(self.features), search=self.search)
        self._last_weights = None
        self.sync_from_model()

    @classmethod
    def from_checkpoint_state(
        cls,
        *,
        policy_state: dict[str, Any],
        model_state: dict[str, Any],
        device: Any,
    ) -> "TDValuePlanningPolicy":
        features = list(policy_state.get("features", []))
        if not features:
            raise ValueError("TD checkpoint missing policy.features")
        search = policy_state.get("search")
        search_obj = HeuristicSearch.model_validate(search) if isinstance(search, dict) else None
        value_model = LinearValueModel(num_features=int(len(features))).to(device=device)
        value_model.load_state_dict(model_state, strict=True)
        policy = cls(features=features, search=search_obj, value_model=value_model)
        policy.load_state_dict(policy_state)
        return policy


__all__ = ["TDValuePlanningPolicy"]
