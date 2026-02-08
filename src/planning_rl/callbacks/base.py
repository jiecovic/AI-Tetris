# src/planning_rl/callbacks/base.py
from __future__ import annotations

from abc import ABC
from typing import Any, Iterable


class PlanningCallback(ABC):
    def __init__(self) -> None:
        self.algo: Any | None = None

    def init_callback(self, algo: Any) -> None:
        self.algo = algo

    def on_start(self, **kwargs: Any) -> None:
        _ = kwargs

    def on_event(self, *, event: str, **kwargs: Any) -> None:
        _ = event
        _ = kwargs

    def on_end(self, **kwargs: Any) -> None:
        _ = kwargs


class CallbackList(PlanningCallback):
    def __init__(self, callbacks: Iterable[PlanningCallback]) -> None:
        super().__init__()
        self.callbacks = list(callbacks)

    def init_callback(self, algo: Any) -> None:
        self.algo = algo
        for cb in self.callbacks:
            cb.init_callback(algo)

    def on_start(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_start(**kwargs)

    def on_event(self, *, event: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_event(event=event, **kwargs)

    def on_end(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_end(**kwargs)


def wrap_callbacks(callbacks: PlanningCallback | Iterable[PlanningCallback] | None) -> PlanningCallback | None:
    if callbacks is None:
        return None
    if isinstance(callbacks, PlanningCallback):
        return callbacks
    return CallbackList(callbacks)


__all__ = ["PlanningCallback", "CallbackList", "wrap_callbacks"]
