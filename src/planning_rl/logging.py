# src/planning_rl/logging.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ScalarLogger(Protocol):
    def log_scalar(self, name: str, value: float, step: int) -> None:
        raise NotImplementedError


class NullLogger:
    def log_scalar(self, name: str, value: float, step: int) -> None:
        _ = name
        _ = value
        _ = step


__all__ = ["NullLogger", "ScalarLogger"]
