# src/planning_rl/policies/vector_param.py
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

from planning_rl.policies.base import PlanningPolicy


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


__all__ = ["VectorParamPolicy"]
