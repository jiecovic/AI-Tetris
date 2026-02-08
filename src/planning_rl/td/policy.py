# src/planning_rl/td/policy.py
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from planning_rl.policies.base import PlanningPolicy


class TDPolicy(PlanningPolicy):
    @property
    @abstractmethod
    def value_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def sync_from_model(self) -> None:
        raise NotImplementedError


__all__ = ["TDPolicy"]
