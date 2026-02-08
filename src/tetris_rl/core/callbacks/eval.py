# src/tetris_rl/core/callbacks/eval.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from tetris_rl.core.callbacks.base import CoreCallback
from tetris_rl.core.training.evaluation.eval_checkpoint_core import (
    EvalCheckpointCore,
    EvalCheckpointCoreSpec,
    EvalRunner,
)


class EvalCallback(CoreCallback):
    def __init__(
        self,
        *,
        spec: EvalCheckpointCoreSpec,
        cfg: Dict[str, Any],
        event: str = "step",
        progress_key: str = "num_timesteps",
        progress_offset: int = 0,
        phase: str = "rl",
        model_getter: Optional[Callable[[Any], Any]] = None,
        extra_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        emit: Optional[Callable[[str], None]] = None,
        log_scalar: Optional[Callable[[str, float, int], None]] = None,
        eval_fn: Optional[EvalRunner] = None,
    ) -> None:
        super().__init__()
        self.event = str(event)
        self.progress_key = str(progress_key)
        self.progress_offset = int(progress_offset)
        self.phase = str(phase)
        self.model_getter = model_getter or (lambda x: x)
        self.extra_metrics_fn = extra_metrics_fn
        self.core = EvalCheckpointCore(
            spec=spec,
            cfg=cfg,
            emit=emit,
            log_scalar=log_scalar,
            eval_fn=eval_fn,
        )

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
        self.core.maybe_tick(
            progress_step=step,
            phase=self.phase,
            model=model,
            extra_metrics_fn=self.extra_metrics_fn,
        )


__all__ = ["EvalCallback"]
