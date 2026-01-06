# src/tetris_rl/game/core/rotation.py
from __future__ import annotations

from tetris_rl.game.core.board import Board
from tetris_rl.game.core.pieceset import PieceSet


def collides(*, board: Board, pieces: PieceSet, kind: str, rot: int, px: int, py: int) -> bool:
    m = pieces.mask(kind, rot)

    h, w = m.shape
    for yy in range(h):
        for xx in range(w):
            if m[yy, xx] == 0:
                continue
            x = px + xx
            y = py + yy
            if x < 0 or x >= board.w or y < 0 or y >= board.h:
                return True
            if board.grid[y, x] != 0:
                return True
    return False


def try_rotate(
        *,
        board: Board,
        pieces: PieceSet,
        kind: str,
        rot: int,
        px: int,
        py: int,
        dir: int,
) -> int:
    """
    Minimal rotation rule: no wall kicks (GB-accuracy later).
    """
    nrot = rot + dir
    if collides(board=board, pieces=pieces, kind=kind, rot=nrot, px=px, py=py):
        return rot
    return nrot
