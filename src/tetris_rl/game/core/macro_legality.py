# src/tetris_rl/game/core/macro_legality.py
from __future__ import annotations

from typing import Optional

import numpy as np

from tetris_rl.game.core.board import Board
from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.game.core.rotation import collides


def macro_illegal_reason_bbox_left(
        *,
        board: Board,
        pieces: PieceSet,
        cache: StaticPlacementCache,
        kind: str,
        rot: int,
        py: int,
        bbox_left_col: int,
) -> Optional[str]:
    """
    Single source of truth for Macro (rot, bbox-left-col) legality at row py.

    North Star legality (macro action is legal iff all hold):
      1) rotation exists for this piece kind (asset-valid rotation)
      2) bbox-left column is within board bounds for that rotation (0..bbox_left_max)
      3) placement at (engine_px, py) does NOT collide with the locked board

    Returns:
      None if legal, else one of:
        - "invalid_rotation"
        - "oob"
        - "collision"
    """
    k = str(kind)
    r = int(rot)
    y = int(py)
    c = int(bbox_left_col)

    # 1) asset-valid rotation
    if not bool(cache.is_valid_rotation(k, r)):
        return "invalid_rotation"

    # 2) bbox-left bounds (geometry / oob)
    try:
        _minx, _bbox_w, bbox_left_max = cache.bbox_params(k, r)
    except Exception:
        # Defensive: treat missing geom as invalid rotation.
        return "invalid_rotation"

    if c < 0 or c > int(bbox_left_max):
        return "oob"

    # 3) collision at current py
    px = int(cache.bbox_left_to_engine_x(k, r, c))
    if bool(collides(board=board, pieces=pieces, kind=k, rot=r, px=px, py=y)):
        return "collision"

    return None


def macro_is_legal_bbox_left(
        *,
        board: Board,
        pieces: PieceSet,
        cache: StaticPlacementCache,
        kind: str,
        rot: int,
        py: int,
        bbox_left_col: int,
) -> bool:
    """
    Convenience wrapper: True iff macro_illegal_reason_bbox_left() returns None.
    """
    return (
            macro_illegal_reason_bbox_left(
                board=board,
                pieces=pieces,
                cache=cache,
                kind=str(kind),
                rot=int(rot),
                py=int(py),
                bbox_left_col=int(bbox_left_col),
            )
            is None
    )


def discrete_action_mask(
        *,
        board: Board,
        pieces: PieceSet,
        cache: StaticPlacementCache,
        kind: str,
        py: int,
) -> np.ndarray:
    """
    Ground-truth Discrete(rot×col) legality mask for the current board + piece at row py.

    Shape: (cache.max_rots * cache.board_w,)

    Indexing:
      rot = aid // board_w
      col = aid % board_w

    Uses bbox-left semantics for col.

    IMPORTANT:
      - This is the SAME concept as "legality" (North Star).
      - Anything else (env, expert, datagen) should reuse this or macro_illegal_reason_bbox_left().
    """
    bw = int(cache.board_w)
    mr = int(cache.max_rots)
    out = np.zeros((mr * bw,), dtype=bool)

    k = str(kind)
    y = int(py)

    for r in range(mr):
        if not bool(cache.is_valid_rotation(k, int(r))):
            continue

        # Only iterate cols that can possibly be legal by geometry.
        try:
            _minx, _bbox_w, bbox_left_max = cache.bbox_params(k, int(r))
        except Exception:
            continue

        for c in range(int(bbox_left_max) + 1):
            if (
                    macro_illegal_reason_bbox_left(
                        board=board,
                        pieces=pieces,
                        cache=cache,
                        kind=k,
                        rot=int(r),
                        py=y,
                        bbox_left_col=int(c),
                    )
                    is None
            ):
                out[int(r) * bw + int(c)] = True

    return out

