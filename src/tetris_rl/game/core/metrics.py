# src/tetris_rl/game/core/metrics.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetris_rl.game.core.constants import EMPTY_CELL


@dataclass(frozen=True)
class BoardSnapshotMetrics:
    """
    Metrics of the LOCKED board only (no active-piece overlay).

    holes:
      empty cells that have at least one occupied cell above in same column
    bumpiness:
      sum(abs(h[i+1] - h[i])) over column heights
    max_height:
      max column height
    agg_height:
      sum of column heights
    """
    holes: int
    bumpiness: int
    max_height: int
    agg_height: int


def board_snapshot_metrics_from_grid(grid: np.ndarray) -> BoardSnapshotMetrics:
    """
    Compute board metrics from a LOCKED board grid.

    Input contract:
      - grid is a 2D numpy array of piece IDs
      - EMPTY_CELL indicates empty

    Performance:
      - no grid copies are made
      - temporary arrays are small (board-sized) and unavoidable for vectorized ops
    """
    _ensure_2d_grid(grid)

    occ = _occ_from_grid(grid)
    heights = _column_heights_from_occ(occ)

    holes = _count_holes_from_occ(occ)
    bump = _bumpiness_from_heights(heights)
    max_h = int(heights.max()) if heights.size > 0 else 0
    agg_h = int(heights.sum()) if heights.size > 0 else 0

    return BoardSnapshotMetrics(
        holes=int(holes),
        bumpiness=int(bump),
        max_height=int(max_h),
        agg_height=int(agg_h),
    )


def board_snapshot_metrics_from_board(board: object) -> BoardSnapshotMetrics:
    """
    Compute board metrics from a Board-like object with `.grid: np.ndarray`.

    This is intentionally structural (no Protocol/Any gymnastics):
      - expects attribute `.grid`
      - expects `.grid` to be a 2D np.ndarray
    """
    g = getattr(board, "grid", None)
    if g is None:
        raise TypeError("board_snapshot_metrics_from_board: expected object with .grid")
    if not isinstance(g, np.ndarray):
        raise TypeError(f"board_snapshot_metrics_from_board: .grid must be np.ndarray, got {type(g).__name__}")
    return board_snapshot_metrics_from_grid(g)


def board_snapshot_metrics_from_game(game: object) -> BoardSnapshotMetrics:
    """
    Compute board metrics from a Game-like object with `.board.grid`.

    This keeps call sites explicit and avoids accidentally passing a render State.
    """
    b = getattr(game, "board", None)
    if b is None:
        raise TypeError("board_snapshot_metrics_from_game: expected object with .board")
    return board_snapshot_metrics_from_board(b)


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def _ensure_2d_grid(grid: np.ndarray) -> None:
    if not isinstance(grid, np.ndarray):
        raise TypeError(f"grid must be np.ndarray, got {type(grid).__name__}")
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape={getattr(grid, 'shape', None)}")


def _occ_from_grid(grid: np.ndarray) -> np.ndarray:
    # bool occupancy; allocates a bool array
    return np.not_equal(grid, EMPTY_CELL)


def _column_heights_from_occ(occ: np.ndarray) -> np.ndarray:
    if occ.ndim != 2:
        raise ValueError(f"occ must be 2D, got shape={occ.shape}")

    h, _w = occ.shape
    any_filled = occ.any(axis=0)

    # argmax returns 0 when all-false; mask those to 0 height
    first_filled = np.argmax(occ, axis=0)
    heights = np.where(any_filled, h - first_filled, 0).astype(np.int64, copy=False)
    return heights


def _count_holes_from_occ(occ: np.ndarray) -> int:
    if occ.ndim != 2:
        raise ValueError(f"occ must be 2D, got shape={occ.shape}")

    # filled_seen[y,x] True if any filled cell exists at or above y in that column
    filled_seen = np.maximum.accumulate(occ, axis=0)
    holes = np.sum((~occ) & filled_seen)
    return int(holes)


def _bumpiness_from_heights(heights: np.ndarray) -> int:
    if heights.ndim != 1:
        raise ValueError(f"heights must be 1D, got shape={heights.shape}")
    if heights.size <= 1:
        return 0
    return int(np.abs(np.diff(heights)).sum())
