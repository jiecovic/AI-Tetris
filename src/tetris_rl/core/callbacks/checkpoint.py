# src/tetris_rl/core/callbacks/checkpoint.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from tetris_rl.core.callbacks.base import CoreCallback


class CheckpointCallback(CoreCallback):
    def __init__(
        self,
        *,
        save_dir: Path,
        every: int = 1,
        prefix: str = "ckpt",
        event: str = "generation_end",
        index_key: str = "generation",
        save_latest: bool = True,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.every = max(1, int(every))
        self.prefix = str(prefix)
        self.event = str(event)
        self.index_key = str(index_key)
        self.save_latest = bool(save_latest)

    def _save(self, path: Path) -> None:
        if self.algo is None:
            return
        save_fn = getattr(self.algo, "save", None)
        if save_fn is None:
            return
        save_fn(path)

    def on_event(self, *, event: str, **kwargs: Any) -> None:
        if event != self.event:
            return
        idx = kwargs.get(self.index_key)
        if idx is None:
            return
        step = int(idx)
        if (step + 1) % self.every != 0:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.save_dir / f"{self.prefix}_{step + 1:04d}.zip"
        self._save(ckpt_path)
        if self.save_latest:
            self._save(self.save_dir / "latest.zip")


__all__ = ["CheckpointCallback"]
