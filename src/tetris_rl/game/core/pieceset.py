# src/tetris_rl/game/core/pieceset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from tetris_rl.utils.paths import pieces_dir


def _parse_color(v: object) -> Optional[Tuple[int, int, int]]:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"color must be a 3-item list/tuple, got {v!r}")
    r, g, b = v
    for c in (r, g, b):
        if not isinstance(c, int) or not (0 <= c <= 255):
            raise ValueError(f"color components must be ints in [0,255], got {v!r}")
    return int(r), int(g), int(b)


def _parse_rotation(rows: Sequence[str]) -> np.ndarray:
    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        raise ValueError("rotation must be a non-empty list of strings")

    width = None
    out: List[List[int]] = []
    for r in rows:
        if not isinstance(r, str) or len(r) == 0:
            raise ValueError(f"rotation rows must be non-empty strings, got {r!r}")
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError(f"rotation rows must have equal width, got widths {width} and {len(r)}")

        out.append([1 if ch == "#" else 0 for ch in r])

    arr = np.asarray(out, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("rotation must form a 2D mask")

    if int(arr.sum()) <= 0:
        raise ValueError("rotation must have at least one filled cell ('#')")
    return arr


@dataclass(frozen=True)
class PieceDef:
    kind: str
    rotations: Tuple[np.ndarray, ...]  # each is (H,W) uint8 mask 0/1
    color: Optional[Tuple[int, int, int]] = None

    def num_rotations(self) -> int:
        return len(self.rotations)

    def mask(self, rot: int) -> np.ndarray:
        rots = self.rotations
        return rots[int(rot) % len(rots)]

    def cell_count(self) -> int:
        # assume all rotations have the same number of blocks (validated)
        return int(self.rotations[0].sum())


@dataclass(frozen=True)
class PieceSet:
    """
    Pure geometry + optional colors, loaded from YAML.

    Provides:
      - stable ordering of kinds (for kind-index mapping)
      - mask(kind, rot)
      - kind_idx(kind) in 0..K-1
      - board_id(kind) in 1..K (for categorical board grids; 0 reserved for empty)

    Asset contract:
      - The rotations listed in YAML for a piece kind are exactly the valid action rotations.
        (So "redundant rotations" should not be present in the YAML.)
    """

    pieces: Dict[str, PieceDef]
    kind_order: Tuple[str, ...]

    @staticmethod
    def default_classic7_path() -> Path:
        return pieces_dir() / "classic7.yaml"

    @classmethod
    def from_yaml(cls, path: Path, *, expected_cells: Optional[int] = None) -> "PieceSet":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError(f"piece YAML must be a mapping at top-level, got {type(data)!r}")

        # Asset-level invariant: number of filled cells per piece
        if expected_cells is None:
            v = data.get("expected_cells", None)
            if isinstance(v, int):
                expected_cells = v
            elif isinstance(v, str):
                expected_cells = int(v)
            elif v is None:
                expected_cells = None
            else:
                raise TypeError(f"expected_cells must be int or str, got {type(v)!r}")

        pieces_node = data.get("pieces")
        if not isinstance(pieces_node, dict) or not pieces_node:
            raise ValueError("piece YAML must contain non-empty mapping 'pieces:'")

        pieces: Dict[str, PieceDef] = {}
        kind_order: List[str] = []

        for kind, spec in pieces_node.items():
            if not isinstance(kind, str) or not kind:
                raise ValueError(f"piece key must be a non-empty string, got {kind!r}")
            if not isinstance(spec, dict):
                raise ValueError(f"piece spec for {kind!r} must be a mapping, got {type(spec)!r}")

            rotations_node = spec.get("rotations")
            if not isinstance(rotations_node, list) or not rotations_node:
                raise ValueError(f"{kind!r}: 'rotations' must be a non-empty list")

            rotations: List[np.ndarray] = []
            for i, rot_rows in enumerate(rotations_node):
                if not isinstance(rot_rows, (list, tuple)):
                    raise ValueError(
                        f"{kind!r}: rotations[{i}] must be a list of strings, got {type(rot_rows)!r}"
                    )
                rotations.append(_parse_rotation(rot_rows))

            cell_counts = [int(r.sum()) for r in rotations]
            if len(set(cell_counts)) != 1:
                raise ValueError(f"{kind!r}: rotations must have same filled cell count, got {cell_counts}")

            if expected_cells is not None and cell_counts[0] != int(expected_cells):
                raise ValueError(f"{kind!r}: expected {expected_cells} filled cells, got {cell_counts[0]}")

            color = _parse_color(spec.get("color"))

            pieces[kind] = PieceDef(kind=kind, rotations=tuple(rotations), color=color)
            kind_order.append(kind)

        return cls(pieces=pieces, kind_order=tuple(kind_order))

    def kinds(self) -> Tuple[str, ...]:
        return self.kind_order

    def __contains__(self, kind: str) -> bool:
        return kind in self.pieces

    def get(self, kind: str) -> PieceDef:
        try:
            return self.pieces[kind]
        except KeyError as e:
            raise KeyError(f"unknown piece kind {kind!r}. known kinds={list(self.kind_order)!r}") from e

    def mask(self, kind: str, rot: int) -> np.ndarray:
        return self.get(kind).mask(rot)

    def num_rotations(self, kind: str) -> int:
        """
        Number of valid rotations for this kind.

        Contract: rotations listed in the YAML are exactly the valid action rotations.
        """
        return int(self.get(kind).num_rotations())

    @staticmethod
    def filled_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Return (minx, miny, maxx, maxy) of non-zero cells.
        If mask has no filled cells, returns (0,0,-1,-1).
        """
        m = np.asarray(mask)
        if m.ndim != 2:
            return (0, 0, -1, -1)
        ys, xs = np.nonzero(m != 0)
        if xs.size == 0:
            return (0, 0, -1, -1)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def bbox_x_range(self, kind: str, rot: int) -> Tuple[int, int, int]:
        """
        Return (minx, maxx, bbox_w) for the filled cells of mask(kind, rot).

        - minx/maxx are in the mask's local coordinates.
        - bbox_w = maxx - minx + 1 (0 if empty mask; should not happen for valid assets).
        """
        m = self.mask(kind, rot)
        minx, _, maxx, _ = self.filled_bbox(m)
        bbox_w = int(max(0, int(maxx) - int(minx) + 1))
        return int(minx), int(maxx), int(bbox_w)

    def kind_idx(self, kind: str) -> int:
        try:
            idx = self.kind_order.index(kind)
        except ValueError as e:
            raise KeyError(f"unknown piece kind {kind!r}") from e
        return int(idx)

    def idx_to_kind(self, idx: int) -> str:
        ii = int(idx)
        if ii < 0 or ii >= len(self.kind_order):
            raise ValueError(f"kind_idx out of range: {ii} (valid 0..{len(self.kind_order) - 1})")
        return self.kind_order[ii]

    def board_id(self, kind: str) -> int:
        return int(self.kind_idx(kind) + 1)

    def board_id_to_kind(self, board_id: int) -> str:
        bid = int(board_id)
        if bid <= 0:
            raise ValueError("board_id must be >= 1 (0 is empty)")
        return self.idx_to_kind(bid - 1)

    def color_of(self, kind: str) -> Optional[Tuple[int, int, int]]:
        return self.get(kind).color

    def bbox_wh(self, kind: str, rot: int) -> Tuple[int, int]:
        m = self.mask(kind, rot)
        ys, xs = np.where(m != 0)
        if len(xs) == 0:
            return 0, 0
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        return w, h

    def max_rotations(self) -> int:
        """
        Maximum number of valid rotations over all kinds in this PieceSet.

        This is derived from the asset YAML (single source of truth).
        Used to define the global discrete action lattice: action_dim = max_rotations * board_w.
        """
        kinds = self.kinds()
        if not kinds:
            return 1
        return max(1, max(self.num_rotations(k) for k in kinds))

    def max_bbox_size(self) -> tuple[int, int]:
        """
        Return (max_bbox_w, max_bbox_h) over all kinds and their valid rotations.
        """
        mw, mh = 0, 0
        for kind in self.kinds():
            n = int(self.num_rotations(kind))
            for r in range(max(1, n)):
                w, h = self.bbox_wh(kind, r)
                mw = max(mw, int(w))
                mh = max(mh, int(h))
        return int(mw), int(mh)

    def max_bbox_height(self) -> int:
        return int(self.max_bbox_size()[1])

    def max_bbox_width(self) -> int:
        return int(self.max_bbox_size()[0])

