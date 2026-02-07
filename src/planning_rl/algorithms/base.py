# src/planning_rl/algorithms/base.py
from __future__ import annotations

from abc import ABC
from typing import Any

from planning_rl.policies import PlanningPolicy


class PlanningAlgorithm(ABC):
    def __init__(self, *, policy: PlanningPolicy) -> None:
        self.policy = policy

    def predict(self, *, env: Any) -> Any:
        return self.policy.predict(env=env)


__all__ = ["PlanningAlgorithm"]
