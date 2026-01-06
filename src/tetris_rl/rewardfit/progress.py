# src/tetris_rl/rewardfit/progress.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass
class RichProgress:
    enabled: bool = True
    mode: str = "shards"  # "none" | "shards" | "states"

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower().strip()

        self._progress = None
        self._task_shards = None
        self._task_states = None

    def __enter__(self) -> "RichProgress":
        if not self.enabled or self.mode == "none":
            return self

        try:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        except Exception:
            # If Rich isn't available for some reason, silently degrade to no-op.
            self.enabled = False
            return self

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=False,
        )
        self._progress.start()

        # tasks are created lazily on first wrap_* call
        self._task_shards = None
        self._task_states = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._progress is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
        self._progress = None
        self._task_shards = None
        self._task_states = None

    def wrap_shards(self, it: Iterable[Any], *, total: int) -> Iterable[Any]:
        if not self.enabled or self.mode == "none" or self._progress is None:
            yield from it
            return

        if self._task_shards is None:
            self._task_shards = self._progress.add_task("shards", total=int(total))

        for x in it:
            yield x
            try:
                self._progress.advance(self._task_shards, 1)
            except Exception:
                pass

    def wrap_states(self, it: Iterable[Any], *, total: int, desc: Optional[str] = None) -> Iterable[Any]:
        """
        Per-shard state progress bar.

        desc is optional to keep the collector ergonomic.
        """
        if not self.enabled or self.mode != "states" or self._progress is None:
            yield from it
            return

        label = str(desc) if desc else "states"

        # reset / recreate each time we wrap a new shard's states
        if self._task_states is not None:
            try:
                self._progress.remove_task(self._task_states)
            except Exception:
                pass
            self._task_states = None

        self._task_states = self._progress.add_task(label, total=int(total))

        for x in it:
            yield x
            try:
                self._progress.advance(self._task_states, 1)
            except Exception:
                pass
