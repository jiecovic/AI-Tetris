# src/planning_rl/policies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class PlanningPolicy(ABC):
    """Policy that selects actions by simulating the environment."""
    @abstractmethod
    def predict(self, *, env: Any) -> Any:
        raise NotImplementedError


class VectorParamPolicy(PlanningPolicy):
    """Planning policy parameterized by a flat vector for optimizers like GA."""
    @property
    @abstractmethod
    def num_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: Sequence[float]) -> None:
        raise NotImplementedError


__all__ = ["PlanningPolicy", "VectorParamPolicy"]
