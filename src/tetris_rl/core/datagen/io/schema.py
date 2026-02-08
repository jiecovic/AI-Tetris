# src/tetris_rl/core/datagen/io/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# BC-only dataset schema (minimal)
# -----------------------------------------------------------------------------

SCHEMA_VERSION = 4

NPZ_GRID = "grid"              # (N,H,W) uint8  cell_id grid: 0 empty, 1..K = kind_idx+1
NPZ_ACTIVE_KIND = "active_kind"  # (N,) uint8  0..K-1
NPZ_NEXT_KIND = "next_kind"      # (N,) uint8  0..K-1
NPZ_ACTION = "action"            # (N,) uint8  0..A-1

REQUIRED_KEYS: Tuple[str, ...] = (
    NPZ_GRID,
    NPZ_ACTIVE_KIND,
    NPZ_NEXT_KIND,
    NPZ_ACTION,
)

OPTIONAL_KEYS: Tuple[str, ...] = tuple()  # BC-only: no optional arrays defined here


def shard_dtypes() -> Dict[str, np.dtype]:
    return {
        NPZ_GRID: np.dtype(np.uint8),
        NPZ_ACTIVE_KIND: np.dtype(np.uint8),
        NPZ_NEXT_KIND: np.dtype(np.uint8),
        NPZ_ACTION: np.dtype(np.uint8),
    }


@dataclass(frozen=True)
class ShardInfo:
    shard_id: int
    file: str  # relative path inside dataset dir (e.g. "shards/shard_0000.npz")
    num_samples: int
    seed: int  # worker/shard seed used for reproducibility
    episode_max_steps: Optional[int] = None  # tolerated, not required


@dataclass(frozen=True)
class DatasetManifest:
    """
    Minimal manifest (JSON) for BC datasets.
    """

    name: str
    created_utc: str

    board_h: int
    board_w: int
    num_kinds: int
    action_dim: int

    schema_version: int = SCHEMA_VERSION

    compression: bool = False

    keys_required: List[str] = field(default_factory=lambda: list(REQUIRED_KEYS))
    dtypes: Dict[str, str] = field(default_factory=dict)

    shards: List[ShardInfo] = field(default_factory=list)


def _require_1d(name: str, arr: np.ndarray, n: int) -> None:
    if arr.ndim != 1 or int(arr.shape[0]) != int(n):
        raise ValueError(f"{name} must be (N,), got shape={arr.shape} vs N={n}")


def validate_shard_arrays(
    *,
    grid: np.ndarray,
    active_kind: np.ndarray,
    next_kind: np.ndarray,
    action: np.ndarray,
    board_h: int,
    board_w: int,
    num_kinds: int,
    action_dim: int,
) -> None:
    """
    Validate shard arrays for BC-only datasets.

    Requirements:
      - grid: (N,H,W) uint8 cell_id in [0..K]
      - active_kind/next_kind: (N,) uint8 in [0..K-1]
      - action: (N,) uint8 in [0..A-1]
    """
    H, W, K, A = int(board_h), int(board_w), int(num_kinds), int(action_dim)

    if H <= 0 or W <= 0:
        raise ValueError(f"invalid board size HxW={H}x{W}")
    if K <= 0:
        raise ValueError(f"invalid num_kinds={K}")
    if A <= 0:
        raise ValueError(f"invalid action_dim={A}")
    if A > 256:
        raise ValueError(f"action_dim={A} > 256 but action is uint8 (must widen action dtype)")

    g = np.asarray(grid)
    ak = np.asarray(active_kind)
    nk = np.asarray(next_kind)
    act = np.asarray(action)

    # shape checks
    if g.ndim != 3:
        raise ValueError(f"{NPZ_GRID} must be (N,H,W), got shape={g.shape}")
    n = int(g.shape[0])
    if int(g.shape[1]) != H or int(g.shape[2]) != W:
        raise ValueError(f"{NPZ_GRID} has (H,W)=({g.shape[1]},{g.shape[2]}) expected ({H},{W})")

    _require_1d(NPZ_ACTIVE_KIND, ak, n)
    _require_1d(NPZ_NEXT_KIND, nk, n)
    _require_1d(NPZ_ACTION, act, n)

    # dtype checks (strict)
    if g.dtype != np.uint8:
        raise ValueError(f"{NPZ_GRID} dtype must be uint8, got {g.dtype}")
    if ak.dtype != np.uint8:
        raise ValueError(f"{NPZ_ACTIVE_KIND} dtype must be uint8, got {ak.dtype}")
    if nk.dtype != np.uint8:
        raise ValueError(f"{NPZ_NEXT_KIND} dtype must be uint8, got {nk.dtype}")
    if act.dtype != np.uint8:
        raise ValueError(f"{NPZ_ACTION} dtype must be uint8, got {act.dtype}")

    # range checks
    if n > 0:
        ak_min, ak_max = int(ak.min()), int(ak.max())
        nk_min, nk_max = int(nk.min()), int(nk.max())
        if ak_min < 0 or ak_max >= K:
            raise ValueError(f"{NPZ_ACTIVE_KIND} out of range: min={ak_min} max={ak_max} K={K}")
        if nk_min < 0 or nk_max >= K:
            raise ValueError(f"{NPZ_NEXT_KIND} out of range: min={nk_min} max={nk_max} K={K}")

        gmin, gmax = int(g.min()), int(g.max())
        if K > 2 and gmax <= 1 and g.size:
            raise ValueError(
                "grid appears binary (max<=1) but K>2. "
                "Must store categorical cell_id grid (0..K) with 0=empty and 1..K=kind_idx+1."
            )
        if gmin < 0 or gmax > K:
            raise ValueError(f"grid out of range: min={gmin} max={gmax} expected within [0..K]={K}")

        amin, amax = int(act.min()), int(act.max())
        if amin < 0 or amax >= A:
            raise ValueError(f"{NPZ_ACTION} out of range: min={amin} max={amax} A={A}")


__all__ = [
    "SCHEMA_VERSION",
    "NPZ_GRID",
    "NPZ_ACTIVE_KIND",
    "NPZ_NEXT_KIND",
    "NPZ_ACTION",
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
    "DatasetManifest",
    "ShardInfo",
    "shard_dtypes",
    "validate_shard_arrays",
]
