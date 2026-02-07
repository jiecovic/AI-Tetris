# src/tetris_rl/core/callbacks/base.py
from __future__ import annotations

from abc import ABC
from typing import Any, Iterable


class CoreCallback(ABC):
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


class CallbackList(CoreCallback):
    def __init__(self, callbacks: Iterable[CoreCallback]) -> None:
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


def wrap_callbacks(callbacks: CoreCallback | Iterable[CoreCallback] | None) -> CoreCallback | None:
    if callbacks is None:
        return None
    if isinstance(callbacks, CoreCallback):
        return callbacks
    return CallbackList(callbacks)


__all__ = ["CoreCallback", "CallbackList", "wrap_callbacks"]
