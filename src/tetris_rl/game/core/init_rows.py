# src/tetris_rl/game/core/init_rows.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InitRowsApplySpec:
    """
    Apply a precomputed bottom-row layout to the locked board.

    Semantics:
      - layout: (R,W) uint8 categorical ids, 0=empty, 1..K=piece-id
      - top_free_rows: (optional) extra cap. Rows above this are guaranteed empty after apply.
        The effective top-free cap is:
            max(top_free_rows, game.spawn_rows + game.pieces.max_bbox_height()).
    """
    layout: np.ndarray
    top_free_rows: int = 0


def _grid_view(board: Any) -> np.ndarray:
    if isinstance(board, np.ndarray):
        arr = board
    else:
        arr = getattr(board, "grid", None)

    if arr is None:
        raise RuntimeError("apply_init_rows_layout requires board.grid (np.ndarray) or an ndarray")

    g = np.asarray(arr)
    if g.ndim != 2:
        raise RuntimeError(f"board grid must be 2D, got shape={g.shape}")
    if not g.flags.writeable:
        raise RuntimeError("board.grid must be writable")
    return g


def _infer_top_free_rows_from_game(game: Any) -> int:
    spawn_rows = int(getattr(game, "spawn_rows", 0))

    pieces = getattr(game, "pieces", None)
    if pieces is None or not hasattr(pieces, "max_bbox_height"):
        raise RuntimeError("apply_init_rows_layout requires game.pieces.max_bbox_height()")

    tetro_h = int(pieces.max_bbox_height())
    return max(0, spawn_rows + tetro_h)


def apply_init_rows_layout(*, game: Any, spec: InitRowsApplySpec) -> None:
    """
    Write `spec.layout` into the bottom rows of game.board.grid.

    Hard guarantees:
      - Never writes into the top-free region.
      - Top-free region is zeroed after apply (prevents lock-out).

    The top-free region is derived from game geometry:
      top_free = max(spec.top_free_rows, game.spawn_rows + max_tetromino_bbox_height)
    """
    board = getattr(game, "board", None)
    if board is None:
        raise RuntimeError("apply_init_rows_layout requires game.board")

    grid = _grid_view(board)
    H, W = int(grid.shape[0]), int(grid.shape[1])

    layout = np.asarray(spec.layout)
    if layout.ndim != 2:
        raise ValueError(f"layout must be 2D (R,W), got shape={layout.shape}")
    if int(layout.shape[1]) != int(W):
        raise ValueError(f"layout width mismatch: layout.W={layout.shape[1]} board.W={W}")

    top_free_game = _infer_top_free_rows_from_game(game)
    top_free_spec = max(0, int(spec.top_free_rows))
    top_free = max(int(top_free_game), int(top_free_spec))

    max_rows = max(0, H - top_free)
    if max_rows <= 0:
        # Still enforce empty top-free (entire board if top_free>=H)
        grid[: min(top_free, H), :] = 0
        return

    R = int(layout.shape[0])
    if R <= 0:
        # Still enforce empty top-free
        if top_free > 0:
            grid[: min(top_free, H), :] = 0
        return

    rows_to_write = min(R, max_rows)

    # Write bottom-most `rows_to_write` rows from the layout (take last rows if layout is taller).
    layout_slice = layout[-rows_to_write:, :].astype(grid.dtype, copy=False)
    grid[H - rows_to_write: H, :] = layout_slice

    # Hard guarantee: keep top-free rows empty.
    if top_free > 0:
        grid[: min(top_free, H), :] = 0


__all__ = ["InitRowsApplySpec", "apply_init_rows_layout"]
