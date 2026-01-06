# src/tetris_rl/datagen/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# NPZ schema (macro-obs aligned; optional reward-fit labels)
# -----------------------------------------------------------------------------

# Bump recommended: FEATURE_NAMES length (F) changed.
SCHEMA_VERSION = 3

# -----------------------------------------------------------------------------
# Base (BC) required keys
# -----------------------------------------------------------------------------
NPZ_GRID = "grid"  # (N,H,W) uint8  cell_id grid: 0 empty, 1..K = kind_idx+1
NPZ_ACTIVE_KIND = "active_kind"  # (N,) uint8  kind_idx 0..K-1 (STRICT)
NPZ_NEXT_KIND = "next_kind"  # (N,) uint8  kind_idx 0..K-1 (STRICT)
NPZ_ACTION = "action"  # (N,) int64  Discrete action_id (rot*w + col)

REQUIRED_KEYS: Tuple[str, ...] = (
    NPZ_GRID,
    NPZ_ACTIVE_KIND,
    NPZ_NEXT_KIND,
    NPZ_ACTION,
)

# -----------------------------------------------------------------------------
# Optional keys (explicitly gated by DataGenSpec.labels)
# -----------------------------------------------------------------------------
# Reward-fit label keys (optional)
NPZ_LEGAL_MASK = "legal_mask"  # (N,A) bool  True if action is legal
NPZ_PHI = "phi"  # (N,A) float32  expert score/logit per action (unnormalized)
NPZ_DELTA = "delta"  # (N,A,F) float32  Δ-features per action

# New engine features (optional, but typically always present)
NPZ_PLACED_CELLS_CLEARED = "placed_cells_cleared"  # (N,) uint8  0..4
NPZ_PLACED_CELLS_ALL_CLEARED = "placed_cells_all_cleared"  # (N,) bool

# Optional metadata arrays (rarely needed; keep optional)
NPZ_ACTION_DIM = "action_dim"  # (1,) int32  stored for convenience/redundancy (optional)
NPZ_FEATURE_NAMES = "feature_names"  # (F,) str  stored for readability (optional)

OPTIONAL_KEYS: Tuple[str, ...] = (
    NPZ_LEGAL_MASK,
    NPZ_PHI,
    NPZ_DELTA,
    NPZ_PLACED_CELLS_CLEARED,
    NPZ_PLACED_CELLS_ALL_CLEARED,
    NPZ_ACTION_DIM,
    NPZ_FEATURE_NAMES,
)

# Canonical feature order for delta labels (F)
#
# IMPORTANT:
# - These define the columns of the reward-fit delta tensor (N,A,F).
# - "placed_cells_*" are included here so reward-fit models_old can interpolate on them.
FEATURE_NAMES: Tuple[str, ...] = (
    "cleared_lines",
    "delta_holes",
    "delta_max_height",
    "delta_bumpiness",
    "delta_agg_height",
    "placed_cells_cleared",
    "placed_cells_all_cleared",
)


def shard_dtypes() -> Dict[str, np.dtype]:
    """
    Declared shard dtypes (writer should cast/validate accordingly).

    Notes:
      - legal_mask is stored as np.bool_ for compactness.
      - feature_names is stored as a unicode string array (no pickle).
    """
    return {
        # base
        NPZ_GRID: np.dtype(np.uint8),
        NPZ_ACTIVE_KIND: np.dtype(np.uint8),
        NPZ_NEXT_KIND: np.dtype(np.uint8),
        NPZ_ACTION: np.dtype(np.int64),
        # optional reward-fit
        NPZ_LEGAL_MASK: np.dtype(np.bool_),
        NPZ_PHI: np.dtype(np.float32),
        NPZ_DELTA: np.dtype(np.float32),
        # optional engine features
        NPZ_PLACED_CELLS_CLEARED: np.dtype(np.uint8),
        NPZ_PLACED_CELLS_ALL_CLEARED: np.dtype(np.bool_),
        # optional metadata
        NPZ_ACTION_DIM: np.dtype(np.int32),
        NPZ_FEATURE_NAMES: np.dtype(np.str_),
    }


@dataclass(frozen=True)
class ShardInfo:
    shard_id: int
    file: str  # relative path inside dataset dir (e.g. "shards/shard_0000.npz")
    num_samples: int
    seed: int  # worker/shard seed used for reproducibility
    episode_max_steps: Optional[int] = None


@dataclass(frozen=True)
class DatasetManifest:
    """
    Minimal manifest (JSON) that makes datasets reusable and self-describing.

    "schema_version" guards interpretation of keys/shapes.
    """

    name: str
    created_utc: str
    schema_version: int = SCHEMA_VERSION

    # environment compatibility / replay metadata
    board_h: int = 20
    board_w: int = 10
    num_kinds: int = 7  # K (number of tetromino kinds)

    # action geometry
    action_dim: int = 40  # typically max_rots * board_w
    max_rots: int = 4

    pieces: str = "classic7"
    piece_rule: str = "k-bag"

    compression: bool = False

    # schema metadata
    keys_required: List[str] = field(default_factory=lambda: list(REQUIRED_KEYS))
    keys_optional: List[str] = field(default_factory=lambda: list(OPTIONAL_KEYS))
    dtypes: Dict[str, str] = field(default_factory=dict)

    # reward-fit metadata (only meaningful if delta/phi/mask are present)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    # generation spec snapshot (best-effort)
    datagen_spec: Dict[str, Any] = field(default_factory=dict)

    shards: List[ShardInfo] = field(default_factory=list)


def _require_1d(name: str, arr: np.ndarray, n: int) -> None:
    if arr.ndim != 1 or int(arr.shape[0]) != int(n):
        raise ValueError(f"{name} must be (N,), got shape={arr.shape} vs N={n}")


def _require_2d(name: str, arr: np.ndarray, n: int, a: int) -> None:
    if arr.ndim != 2 or int(arr.shape[0]) != int(n) or int(arr.shape[1]) != int(a):
        raise ValueError(f"{name} must be (N,A), got shape={arr.shape} vs (N,A)=({n},{a})")


def _require_3d(name: str, arr: np.ndarray, n: int, a: int, f: int) -> None:
    if (
            arr.ndim != 3
            or int(arr.shape[0]) != int(n)
            or int(arr.shape[1]) != int(a)
            or int(arr.shape[2]) != int(f)
    ):
        raise ValueError(f"{name} must be (N,A,F), got shape={arr.shape} vs (N,A,F)=({n},{a},{f})")


def validate_shard_arrays(
        *,
        grid: np.ndarray,
        active_kind: np.ndarray,
        next_kind: np.ndarray,
        action: np.ndarray,
        board_h: int,
        board_w: int,
        num_kinds: int,
        # optional engine features
        placed_cells_cleared: Optional[np.ndarray] = None,
        placed_cells_all_cleared: Optional[np.ndarray] = None,
        # optional reward-fit
        legal_mask: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
        delta: Optional[np.ndarray] = None,
        # optional metadata
        action_dim: Optional[int] = None,
        feature_names: Optional[np.ndarray] = None,
) -> None:
    """
    Validate shapes/dtypes before writing.

    Base requirements:
      - grid: (N,H,W) uint8 cell_id in [0..K], where 0=empty and 1..K=kind_idx+1
      - active_kind/next_kind: (N,) kind_idx in [0..K-1]
      - action: (N,) int64 in [0..A-1] if action_dim provided
    """
    H, W, K = int(board_h), int(board_w), int(num_kinds)
    if H <= 0 or W <= 0:
        raise ValueError(f"invalid board size HxW={H}x{W}")
    if K <= 0:
        raise ValueError(f"invalid num_kinds K={K}")

    if grid.ndim != 3:
        raise ValueError(f"{NPZ_GRID} must be (N,H,W), got shape={grid.shape}")
    n = int(grid.shape[0])
    if int(grid.shape[1]) != H or int(grid.shape[2]) != W:
        raise ValueError(f"{NPZ_GRID} has (H,W)=({grid.shape[1]},{grid.shape[2]}) expected ({H},{W})")

    _require_1d(NPZ_ACTIVE_KIND, np.asarray(active_kind), n)
    _require_1d(NPZ_NEXT_KIND, np.asarray(next_kind), n)
    _require_1d(NPZ_ACTION, np.asarray(action), n)

    if n > 0:
        ak = np.asarray(active_kind)
        nk = np.asarray(next_kind)
        if int(ak.min()) < 0 or int(ak.max()) >= K:
            raise ValueError(f"{NPZ_ACTIVE_KIND} out of range: min={int(ak.min())} max={int(ak.max())} K={K}")
        if int(nk.min()) < 0 or int(nk.max()) >= K:
            raise ValueError(f"{NPZ_NEXT_KIND} out of range: min={int(nk.min())} max={int(nk.max())} K={K}")

        g = np.asarray(grid)
        mx = int(g.max()) if g.size else 0
        if K > 2 and mx <= 1 and g.size:
            raise ValueError(
                "grid appears binary (max<=1) but K>2. "
                "Macro datasets must store categorical cell_id grid (0..K) with 0=empty and 1..K=kind_idx+1."
            )
        if int(g.min()) < 0 or int(mx) > K:
            raise ValueError(f"grid out of range: min={int(g.min())} max={mx} expected within [0..K]={K}")

    # ------------------------------------------------------------
    # Optional engine features
    # ------------------------------------------------------------
    if placed_cells_cleared is not None:
        _require_1d(NPZ_PLACED_CELLS_CLEARED, np.asarray(placed_cells_cleared), n)
        pcc = np.asarray(placed_cells_cleared)
        if pcc.size:
            mn, mx = int(pcc.min()), int(pcc.max())
            if mn < 0 or mx > 4:
                raise ValueError(
                    f"{NPZ_PLACED_CELLS_CLEARED} out of range: min={mn} max={mx} expected within [0..4]"
                )
    if placed_cells_all_cleared is not None:
        _require_1d(NPZ_PLACED_CELLS_ALL_CLEARED, np.asarray(placed_cells_all_cleared), n)

    # ------------------------------------------------------------
    # Reward-fit invariants: ALL-or-NONE (schema-level)
    # ------------------------------------------------------------
    rf = (legal_mask, phi, delta)
    any_rf = any(x is not None for x in rf)
    all_rf = all(x is not None for x in rf)
    if any_rf and not all_rf:
        raise ValueError("reward-fit arrays must be ALL present or ALL absent (legal_mask, phi, delta)")

    if any_rf:
        A = int(action_dim) if action_dim is not None else None
        if A is None or A <= 0:
            raise ValueError("reward-fit arrays present but action_dim is missing/invalid")

        if legal_mask is not None:
            _require_2d(NPZ_LEGAL_MASK, np.asarray(legal_mask), n, A)
        if phi is not None:
            _require_2d(NPZ_PHI, np.asarray(phi), n, A)

        fn = feature_names
        if fn is None:
            f = int(len(FEATURE_NAMES))
        else:
            fn_arr = np.asarray(fn)
            if fn_arr.ndim != 1:
                raise ValueError(f"{NPZ_FEATURE_NAMES} must be (F,), got shape={fn_arr.shape}")
            f = int(fn_arr.shape[0])

        if delta is not None:
            _require_3d(NPZ_DELTA, np.asarray(delta), n, A, f)

        a = np.asarray(action)
        if a.size:
            if int(a.min()) < 0 or int(a.max()) >= A:
                raise ValueError(f"{NPZ_ACTION} out of range: min={int(a.min())} max={int(a.max())} A={A}")

    else:
        if action_dim is not None and int(action_dim) > 0:
            A = int(action_dim)
            a = np.asarray(action)
            if a.size:
                if int(a.min()) < 0 or int(a.max()) >= A:
                    raise ValueError(f"{NPZ_ACTION} out of range: min={int(a.min())} max={int(a.max())} A={A}")


__all__ = [
    "SCHEMA_VERSION",
    "NPZ_GRID",
    "NPZ_ACTIVE_KIND",
    "NPZ_NEXT_KIND",
    "NPZ_ACTION",
    "NPZ_LEGAL_MASK",
    "NPZ_PHI",
    "NPZ_DELTA",
    "NPZ_PLACED_CELLS_CLEARED",
    "NPZ_PLACED_CELLS_ALL_CLEARED",
    "NPZ_ACTION_DIM",
    "NPZ_FEATURE_NAMES",
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
    "FEATURE_NAMES",
    "DatasetManifest",
    "ShardInfo",
    "shard_dtypes",
    "validate_shard_arrays",
]
