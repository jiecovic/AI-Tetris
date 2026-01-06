# src/tetris_rl/game/core/placement_cache.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.game.core.types import ActivePiece


@dataclass(frozen=True)
class StaticPlacementCache:
    """
    Asset-dependent + board-width-dependent *GEOMETRY* cache for macro placements.

    Important:
      - This cache is intentionally board-content-agnostic.
      - It supports only the static parts of legality:
          (a) rotation exists for the piece kind (asset-valid)
          (b) bbox-left column is within [0..bbox_left_max] (OOB/geometry)

    Full macro legality additionally requires a board-dependent collision check
    and MUST live outside this module (see game/core/macro_legality.py).

    Contract:
      - PieceSet YAML rotations are the valid action rotations per kind.
      - bbox-left placement semantics:
          bbox_left_col is the board column of the LEFTMOST filled cell in the rotated mask.
          engine px (top-left of mask) is px = bbox_left_col - minx.

    Stores per (kind, rot):
      - minx: leftmost filled x in mask coords
      - bbox_w: width of filled bbox in mask coords
      - bbox_left_max: max legal bbox-left col on the board: board_w - bbox_w
      - valid_rot: whether rot is a valid rotation for this kind (rot < num_rotations(kind))

    Action-space geometry:
      - The discrete macro action space is defined as rot×col with shape:
            action_dim = max_rots * board_w
      - max_rots is derived from the PieceSet assets (single source of truth):
            max_rots = pieces.max_rotations()
        (i.e., maximum number of rotations across all kinds in the asset YAML).
    """

    board_w: int
    max_rots: int
    kinds: Tuple[str, ...]

    # (kind, rot) -> (minx, bbox_w, bbox_left_max)
    geom: Dict[Tuple[str, int], Tuple[int, int, int]]

    # (kind, rot) -> bool
    valid_rot: Dict[Tuple[str, int], bool]

    # kind -> n_rots (valid rotations)
    n_rots: Dict[str, int]

    @classmethod
    def build(cls, *, pieces: PieceSet, board_w: int) -> "StaticPlacementCache":
        bw = int(board_w)
        if bw <= 0:
            raise ValueError(f"board_w must be positive, got {bw}")

        # Single source of truth: derive from assets.
        try:
            mr = int(pieces.max_rotations())
        except Exception as e:
            raise RuntimeError("PieceSet must provide max_rotations() derived from YAML rotations") from e

        if mr <= 0:
            raise RuntimeError(f"invalid max_rots derived from assets: {mr}")

        kinds = tuple(str(k) for k in pieces.kinds())

        geom: Dict[Tuple[str, int], Tuple[int, int, int]] = {}
        valid_rot: Dict[Tuple[str, int], bool] = {}
        n_rots: Dict[str, int] = {}

        for kind in kinds:
            nr = int(pieces.num_rotations(kind))
            nr = max(1, nr)
            n_rots[kind] = nr

            for r in range(mr):
                is_valid = bool(r < nr)
                valid_rot[(kind, int(r))] = is_valid
                if not is_valid:
                    continue

                minx, _maxx, bbox_w = pieces.bbox_x_range(kind, int(r))
                bbox_w = int(bbox_w)
                if bbox_w <= 0:
                    # defensive fallback; should never happen for valid assets
                    minx, bbox_w = 0, 1

                bbox_left_max = max(0, bw - bbox_w)
                geom[(kind, int(r))] = (int(minx), int(bbox_w), int(bbox_left_max))

        return cls(
            board_w=bw,
            max_rots=mr,
            kinds=kinds,
            geom=geom,
            valid_rot=valid_rot,
            n_rots=n_rots,
        )

    def is_valid_rotation(self, kind: str, rot: int) -> bool:
        return bool(self.valid_rot.get((str(kind), int(rot)), False))

    def bbox_params(self, kind: str, rot: int) -> Tuple[int, int, int]:
        """
        Return (minx, bbox_w, bbox_left_max) for (kind, rot).
        Raises KeyError if rot is invalid for this kind.
        """
        key = (str(kind), int(rot))
        if not self.is_valid_rotation(kind, rot):
            raise KeyError(f"invalid rotation: kind={kind!r}, rot={int(rot)}")
        try:
            return self.geom[key]
        except KeyError as e:
            raise KeyError(f"missing geom cache for kind={kind!r}, rot={int(rot)}") from e

    def bbox_left_to_engine_x(self, kind: str, rot: int, bbox_left_col: int) -> int:
        """
        Convert bbox-left board column to engine px (mask top-left x).
        """
        minx, _bbox_w, _bbox_left_max = self.bbox_params(kind, rot)
        return int(int(bbox_left_col) - int(minx))

    def is_geom_legal_bbox_left(self, *, kind: str, rot: int, bbox_left_col: int) -> bool:
        """
        GEOMETRY-only legality (no collision):
          - rotation exists for kind
          - bbox_left_col in [0..bbox_left_max]
        """
        k = str(kind)
        r = int(rot)
        if not self.is_valid_rotation(k, r):
            return False
        _minx, _bbox_w, bbox_left_max = self.bbox_params(k, r)
        c = int(bbox_left_col)
        return bool(0 <= c <= int(bbox_left_max))

    def discrete_geom_mask(self, *, kind: str) -> np.ndarray:
        """
        GEOMETRY-only Discrete(rot×col) mask of shape (max_rots * board_w).

        True means:
          - rotation exists
          - col is within [0..bbox_left_max]
        """
        bw = int(self.board_w)
        mr = int(self.max_rots)
        out = np.zeros((mr * bw,), dtype=bool)

        k = str(kind)
        for r in range(mr):
            if not self.is_valid_rotation(k, int(r)):
                continue
            _minx, _bbox_w, bbox_left_max = self.bbox_params(k, int(r))
            for c in range(int(bbox_left_max) + 1):
                out[int(r) * bw + int(c)] = True

        return out


def apply_bbox_left_placement(
        *,
        active: ActivePiece,
        kind: str,
        rot: int,
        bbox_left_col: int,
        cache: StaticPlacementCache,
) -> ActivePiece:
    """
    Return a new ActivePiece with (kind, rot, x) set using bbox-left semantics,
    keeping y from the provided active piece.
    """
    px = cache.bbox_left_to_engine_x(str(kind), int(rot), int(bbox_left_col))
    return ActivePiece(kind=str(kind), rot=int(rot), x=int(px), y=int(active.y))


__all__ = [
    "StaticPlacementCache",
    "apply_bbox_left_placement",
]
