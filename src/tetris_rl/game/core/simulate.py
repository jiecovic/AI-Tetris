# src/tetris_rl/game/core/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from tetris_rl.game.core.board import Board
from tetris_rl.game.core.metrics import BoardSnapshotMetrics, board_snapshot_metrics_from_grid
from tetris_rl.game.core.macro_legality import macro_illegal_reason_bbox_left
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.game.core.rotation import collides


def _clear_full_lines_grid(grid: np.ndarray, *, empty_cell: int) -> tuple[np.ndarray, int]:
    """
    Pure numpy line-clear. Returns (new_grid, cleared).
    Does NOT mutate the input.
    """
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape={grid.shape}")

    full = np.all(grid != empty_cell, axis=1)
    cleared = int(full.sum())
    if cleared <= 0:
        return grid, 0

    kept = grid[~full]
    new_rows = np.zeros((cleared, grid.shape[1]), dtype=grid.dtype)
    out = np.vstack([new_rows, kept])
    return out, cleared


def _cleared_row_indices(grid_locked: np.ndarray, *, empty_cell: int) -> np.ndarray:
    """
    Return 1D int array of row indices that are full (would be cleared),
    based on the LOCKED grid before clearing.
    """
    if grid_locked.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape={grid_locked.shape}")
    full = np.all(grid_locked != empty_cell, axis=1)
    return np.flatnonzero(full).astype(np.int32, copy=False)


@dataclass(frozen=True)
class SimPlacementResult:
    """
    Result of simulating a macro placement on a LOCKED grid.

    Contract (STRICT, matches MacroTetrisEnv):
      A placement is legal iff:
        - rotation exists for the piece kind
        - bbox-left col is within [0..bbox_left_max] for that (kind,rot)
        - does NOT collide with the locked board at start_y

    Notes:
      - grid_after is a NEW array (copy) containing the post-lock, post-clear board.
      - cleared_lines is number of cleared rows.
      - game_over indicates engine-style lock-out after the placement (spawn_rows check),
        OR an illegal placement (strict legality failed).
      - illegal_reason is set when strict legality failed (invalid_rotation/oob/collision).
      - metrics_after are computed from grid_after (LOCKED only).

    Extra signals:
      - placed_cells_cleared: how many of the placed tetromino cells (<=4) vanished due to cleared lines (0..4)
      - placed_cells_all_cleared: True iff all placed tetromino cells vanished due to cleared lines
    """
    grid_after: np.ndarray
    cleared_lines: int
    game_over: bool
    metrics_after: BoardSnapshotMetrics
    placed_cells_cleared: int
    placed_cells_all_cleared: bool
    illegal_reason: Optional[str] = None


class MacroPlacementSimulator:
    """
    Pure, side-effect-free macro placement simulator.

    STRICT semantics (must match MacroTetrisEnv):
      - rot is NOT wrapped
      - col is bbox-left column, NOT clamped
      - engine-x (mask top-left) = bbox_left_col - minx
      - start_y is the CURRENT active-piece y (or 0 for newly spawned pieces)
    """

    def __init__(
            self,
            *,
            pieces: Any,
            board_w: int,
            board_h: int,
            spawn_rows: int = 2,
            empty_cell: int = 0,
    ) -> None:
        self.pieces = pieces
        self.w = int(board_w)
        self.h = int(board_h)
        self.spawn_rows = int(spawn_rows)
        self.empty_cell = int(empty_cell)

        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"invalid board size (h={self.h}, w={self.w})")

        if not hasattr(self.pieces, "max_rotations"):
            raise RuntimeError("pieces must provide max_rotations() derived from YAML rotations")

        self.max_rots = int(self.pieces.max_rotations())
        if self.max_rots <= 0:
            raise RuntimeError(f"invalid max_rots derived from assets: {self.max_rots}")

        self._legal = StaticPlacementCache.build(
            pieces=self.pieces,
            board_w=self.w,
        )

    def _illegal_result(self, *, grid: np.ndarray, reason: str) -> SimPlacementResult:
        g0 = np.asarray(grid)
        metrics0 = board_snapshot_metrics_from_grid(g0)
        return SimPlacementResult(
            grid_after=np.array(g0, copy=True),
            cleared_lines=0,
            game_over=True,
            metrics_after=metrics0,
            placed_cells_cleared=0,
            placed_cells_all_cleared=False,
            illegal_reason=str(reason),
        )

    def simulate(
            self,
            *,
            grid: np.ndarray,
            kind: str,
            rot: int,
            col: int,
            start_y: int,
    ) -> SimPlacementResult:
        g0 = np.asarray(grid)
        if g0.ndim != 2:
            raise ValueError(f"grid must be 2D, got shape={getattr(g0, 'shape', None)}")
        if int(g0.shape[0]) != self.h or int(g0.shape[1]) != self.w:
            raise ValueError(f"grid shape must be (h={self.h}, w={self.w}), got {g0.shape}")

        k = str(kind)
        r = int(rot)
        c = int(col)
        y0 = int(start_y)

        board0 = Board(h=self.h, w=self.w, grid=g0)

        # ------------------------------------------------------------
        # STRICT legality checks (single ground truth)
        # ------------------------------------------------------------
        reason = macro_illegal_reason_bbox_left(
            board=board0,
            pieces=self.pieces,
            cache=self._legal,
            kind=k,
            rot=r,
            py=y0,
            bbox_left_col=c,
        )
        if reason is not None:
            return self._illegal_result(grid=g0, reason=str(reason))

        x_engine = int(self._legal.bbox_left_to_engine_x(k, r, c))

        # ------------------------------------------------------------
        # Hard drop from start_y (side-effect-free)
        # ------------------------------------------------------------
        y = int(y0)
        while True:
            if collides(board=board0, pieces=self.pieces, kind=k, rot=r, px=int(x_engine), py=int(y + 1)):
                break
            y += 1
            if y > self.h + 4:
                break

        g1 = np.array(g0, copy=True)
        placed_cells = self._lock_into_grid(g1, kind=k, rot=r, px=int(x_engine), py=int(y))

        # Determine cleared rows from the locked (pre-clear) grid
        cleared_rows = _cleared_row_indices(g1, empty_cell=self.empty_cell)
        cleared_set = set(int(x) for x in cleared_rows.tolist())

        placed_cells_cleared = 0
        if placed_cells and cleared_set:
            placed_cells_cleared = int(sum(1 for (_x, yy) in placed_cells if int(yy) in cleared_set))
        placed_cells_all_cleared = bool(placed_cells and placed_cells_cleared == len(placed_cells))

        g2, cleared = _clear_full_lines_grid(g1, empty_cell=self.empty_cell)
        game_over = self._is_lock_out(g2)

        metrics = board_snapshot_metrics_from_grid(g2)
        return SimPlacementResult(
            grid_after=g2,
            cleared_lines=int(cleared),
            game_over=bool(game_over),
            metrics_after=metrics,
            placed_cells_cleared=int(placed_cells_cleared),
            placed_cells_all_cleared=bool(placed_cells_all_cleared),
            illegal_reason=None,
        )

    def _is_lock_out(self, grid_after: np.ndarray) -> bool:
        sr = int(self.spawn_rows)
        if sr <= 0:
            return False
        top = grid_after[:sr, :]
        return bool(np.any(top != self.empty_cell))

    def _mask(self, kind: str, rot: int) -> np.ndarray:
        return np.asarray(self.pieces.mask(kind, rot))

    def _lock_into_grid(self, grid: np.ndarray, *, kind: str, rot: int, px: int, py: int) -> Sequence[Tuple[int, int]]:
        """
        Lock the piece into `grid` (in-place). Returns placed cell coordinates [(x,y), ...]
        for cells that actually landed inside the board bounds.
        """
        m = self._mask(kind, rot)
        filled_id = 1

        placed: list[tuple[int, int]] = []

        mh, mw = m.shape
        for yy in range(mh):
            for xx in range(mw):
                if int(m[yy, xx]) == 0:
                    continue
                x = int(px + xx)
                y = int(py + yy)
                if 0 <= x < self.w and 0 <= y < self.h:
                    grid[y, x] = filled_id
                    placed.append((int(x), int(y)))

        return placed


__all__ = [
    "SimPlacementResult",
    "MacroPlacementSimulator",
]
