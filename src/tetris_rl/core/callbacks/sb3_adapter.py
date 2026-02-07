# src/tetris_rl/core/callbacks/sb3_adapter.py
from __future__ import annotations

from typing import Iterable

from stable_baselines3.common.callbacks import BaseCallback

from tetris_rl.core.callbacks.base import CoreCallback, wrap_callbacks


class SB3CallbackAdapter(BaseCallback):
    """
    Adapt CoreCallback events to SB3's callback lifecycle.
    """

    def __init__(self, callback: CoreCallback | Iterable[CoreCallback], verbose: int = 0) -> None:
        super().__init__(verbose=int(verbose))
        self.core = wrap_callbacks(callback)

    def _init_callback(self) -> None:
        if self.core is not None:
            self.core.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.core is not None:
            self.core.on_start(num_timesteps=int(self.num_timesteps))

    def _on_step(self) -> bool:
        if self.core is not None:
            self.core.on_event(event="step", num_timesteps=int(self.num_timesteps))
        return True

    def _on_training_end(self) -> None:
        if self.core is not None:
            self.core.on_end(num_timesteps=int(self.num_timesteps))


__all__ = ["SB3CallbackAdapter"]
