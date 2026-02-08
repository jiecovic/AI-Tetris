# src/tetris_rl/core/callbacks/planning_adapter.py
from __future__ import annotations

from typing import Any, Iterable

from planning_rl.callbacks import PlanningCallback
from tetris_rl.core.callbacks.base import CoreCallback, wrap_callbacks


class PlanningCallbackAdapter(PlanningCallback):
    def __init__(self, callback: CoreCallback | Iterable[CoreCallback]) -> None:
        super().__init__()
        self.core = wrap_callbacks(callback)

    def init_callback(self, algo: Any) -> None:
        self.algo = algo
        if self.core is not None:
            self.core.init_callback(algo)

    def on_start(self, **kwargs: Any) -> None:
        if self.core is not None:
            self.core.on_start(**kwargs)

    def on_event(self, *, event: str, **kwargs: Any) -> None:
        if self.core is not None:
            self.core.on_event(event=event, **kwargs)

    def on_end(self, **kwargs: Any) -> None:
        if self.core is not None:
            self.core.on_end(**kwargs)


__all__ = ["PlanningCallbackAdapter"]
