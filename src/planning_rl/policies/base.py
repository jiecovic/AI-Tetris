# src/planning_rl/policies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PlanningPolicy(ABC):
    """Policy that selects actions by simulating the environment."""

    @abstractmethod
    def predict(self, *, env: Any) -> Any:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state


__all__ = ["PlanningPolicy"]
