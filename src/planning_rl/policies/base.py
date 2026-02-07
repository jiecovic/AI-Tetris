# src/planning_rl/policies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class PlanningPolicy(ABC):
    """Policy that selects actions by simulating the environment."""
    @abstractmethod
    def predict(self, *, env: Any) -> Any:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state


class VectorParamPolicy(PlanningPolicy):
    """Planning policy parameterized by a flat vector for optimizers like GA."""
    @property
    @abstractmethod
    def num_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> Sequence[float]:
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: Sequence[float]) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {"params": list(self.get_params())}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        params = state.get("params")
        if params is None:
            return
        self.set_params(params)


__all__ = ["PlanningPolicy", "VectorParamPolicy"]
