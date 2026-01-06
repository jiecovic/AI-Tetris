# src/tetris_rl/game/core/board.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Board:
    h: int
    w: int
    grid: np.ndarray  # locked blocks only (0=empty, >=1 piece ids)

    @classmethod
    def empty(cls, *, h: int, w: int) -> "Board":
        return cls(h=h, w=w, grid=np.zeros((h, w), dtype=np.uint8))

    def clear_full_lines(self) -> int:
        full = np.all(self.grid != 0, axis=1)
        cleared = int(full.sum())
        if cleared <= 0:
            return 0
        self.grid = self.grid[~full]
        new_rows = np.zeros((cleared, self.w), dtype=np.uint8)
        self.grid = np.vstack([new_rows, self.grid])
        return cleared
