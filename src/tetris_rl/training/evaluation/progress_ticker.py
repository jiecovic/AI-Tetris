# src/tetris_rl/training/evaluation/progress_ticker.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgressTicker:
    """
    Generic "tick every N units" helper.

    Robust to progress jumps:
      - ticks when progress >= next_tick_at
      - advances next_tick_at to the next multiple boundary

    Works for:
      - SB3 timesteps (env steps)
      - BC samples_seen
      - BC updates
    """

    every: int
    last_tick_at: int = 0
    next_tick_at: int = 0

    def __post_init__(self) -> None:
        e = int(self.every)
        if e < 0:
            raise ValueError(f"every must be >= 0, got {e}")
        self.every = e
        self.last_tick_at = int(self.last_tick_at)
        self.next_tick_at = int(self.next_tick_at) if int(self.next_tick_at) > 0 else max(1, int(e)) if e > 0 else 0

    def init_from_progress(self, progress: int) -> None:
        """
        Initialize next tick boundary from an existing progress value.
        """
        t0 = int(progress)
        e = int(self.every)
        if e <= 0:
            self.next_tick_at = 0
            return
        self.next_tick_at = max(e, ((t0 // e) + 1) * e)

    def should_tick(self, progress: int) -> bool:
        e = int(self.every)
        if e <= 0:
            return False
        t = int(progress)
        if t <= 0:
            return False
        if t == int(self.last_tick_at):
            return False
        return t >= int(self.next_tick_at)

    def mark_ticked(self, progress: int) -> None:
        t = int(progress)
        self.last_tick_at = t

        e = int(self.every)
        if e <= 0:
            self.next_tick_at = 0
            return
        self.next_tick_at = ((t // e) + 1) * e
