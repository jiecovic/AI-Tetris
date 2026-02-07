# src/tetris_rl/core/callbacks/latest.py
from __future__ import annotations

from typing import Any, Callable, Optional

from tetris_rl.core.callbacks.base import CoreCallback
from tetris_rl.core.training.evaluation.latest_checkpoint_core import (
    LatestCheckpointCore,
    LatestCheckpointCoreSpec,
)


class LatestCallback(CoreCallback):
    def __init__(
        self,
        *,
        spec: LatestCheckpointCoreSpec,
        event: str = "step",
        progress_key: str = "num_timesteps",
        progress_offset: int = 0,
        model_getter: Optional[Callable[[Any], Any]] = None,
        emit: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__()
        self.event = str(event)
        self.progress_key = str(progress_key)
        self.progress_offset = int(progress_offset)
        self.model_getter = model_getter or (lambda x: x)
        self.core = LatestCheckpointCore(spec=spec, emit=emit)

    def on_start(self, **kwargs: Any) -> None:
        progress = int(kwargs.get(self.progress_key, 0))
        self.core.init(progress_step=progress)

    def on_event(self, *, event: str, **kwargs: Any) -> None:
        if event != self.event:
            return
        progress = kwargs.get(self.progress_key)
        if progress is None:
            return
        step = int(progress) + int(self.progress_offset)
        model = self.model_getter(self.algo)
        self.core.maybe_tick(progress_step=step, model=model)


__all__ = ["LatestCallback"]
