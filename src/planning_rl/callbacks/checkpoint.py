# src/planning_rl/callbacks/checkpoint.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from planning_rl.callbacks.base import PlanningCallback


class CheckpointCallback(PlanningCallback):
    def __init__(
        self,
        *,
        save_dir: Path,
        every: int = 1,
        prefix: str = "ga",
        event: str = "generation_end",
        save_latest: bool = True,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.every = max(1, int(every))
        self.prefix = str(prefix)
        self.event = str(event)
        self.save_latest = bool(save_latest)

    def on_event(self, *, event: str, **kwargs: Any) -> None:
        if event != self.event:
            return
        if self.algo is None:
            return
        gen = kwargs.get("generation")
        if gen is None:
            return
        gen = int(gen)
        if (gen + 1) % self.every != 0:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.save_dir / f"{self.prefix}_gen_{gen + 1:04d}.zip"
        self.algo.save(ckpt_path)
        if self.save_latest:
            latest_path = self.save_dir / "latest.zip"
            self.algo.save(latest_path)


__all__ = ["CheckpointCallback"]
