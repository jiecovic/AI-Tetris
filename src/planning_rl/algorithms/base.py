# src/planning_rl/algorithms/base.py
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Generic, TypeVar

from planning_rl.policies import PlanningPolicy

PolicyT = TypeVar("PolicyT", bound=PlanningPolicy)


class PlanningAlgorithm(Generic[PolicyT], ABC):
    def __init__(self, *, policy: PolicyT) -> None:
        self.policy: PolicyT = policy

    def predict(self, *, env: Any) -> Any:
        return self.policy.predict(env=env)

    def save(self, path: Path) -> Path:
        raise NotImplementedError


__all__ = ["PlanningAlgorithm"]
