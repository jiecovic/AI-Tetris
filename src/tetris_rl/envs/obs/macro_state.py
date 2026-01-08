# src/tetris_rl/env_bundles/obs/macro_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class MacroObsSpec:
    """
    Canonical macro raw observation spec.

    obs = Dict({
      "grid":        Box(low=0, high=K, shape=(H,W), dtype=uint8),   # 0 empty, 1..K piece ids (categorical)
      "active_kind": Discrete(K),                                    # 0..K-1 (STRICT kind indices)
      "next_kind":   Discrete(K),                                    # 0..K-1 (STRICT kind indices)
    })

    Contract:
      - grid is categorical piece-id ALWAYS:
          0 = empty
          1..K = kind_idx+1
      - active_kind / next_kind are STRICT kind indices 0..K-1 from State (or dict snapshot).
    """
    board_h: int
    board_w: int
    num_kinds: int


def build_macro_obs_space(*, spec: MacroObsSpec) -> spaces.Dict:
    H, W, K = int(spec.board_h), int(spec.board_w), int(spec.num_kinds)
    if H <= 0 or W <= 0:
        raise ValueError(f"invalid board size HxW={H}x{W}")
    if K <= 0:
        raise ValueError(f"invalid num_kinds K={K}")

    return spaces.Dict(
        {
            "grid": spaces.Box(low=0, high=int(K), shape=(H, W), dtype=np.uint8),
            "active_kind": spaces.Discrete(int(K)),
            "next_kind": spaces.Discrete(int(K)),
        }
    )


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if _is_mapping(obj):
        # mypy: Mapping[str, Any]
        return obj.get(key, default)  # type: ignore[call-arg]
    return getattr(obj, key, default)


def _extract_locked_grid(game: Any, state: Any) -> np.ndarray:
    """
    Prefer game.board.grid (authoritative locked board) if present (legacy path).
    Otherwise use state["grid"] / state.grid.

    With Rust engine snapshots, we expect `state["grid"]` already contains the locked grid
    (typically visible 20x10).
    """
    if game is not None:
        board = getattr(game, "board", None)
        if board is not None:
            g = getattr(board, "grid", None)
            if g is not None:
                return np.asarray(g)

    g2 = _get_field(state, "grid", None)
    if g2 is None:
        raise RuntimeError("cannot extract locked grid: neither game.board.grid nor state['grid']/state.grid is available")
    return np.asarray(g2)


def _extract_active_kind_idx(state: Any) -> int:
    v = _get_field(state, "active_kind_idx", None)
    if v is None:
        raise RuntimeError("cannot extract active_kind_idx: missing (strict contract)")
    return int(v)


def _extract_next_kind_idx(state: Any) -> int:
    v = _get_field(state, "next_kind_idx", None)
    if v is None:
        raise RuntimeError("cannot extract next_kind_idx: missing (strict contract)")
    return int(v)


def encode_macro_obs(*, game: Any, state: Any, spec: MacroObsSpec) -> Dict[str, Any]:
    """
    Encode the canonical raw obs dict (North Star).

    Works with:
      - legacy Python state objects (attributes)
      - Rust PyO3 snapshots (dict-like)

    Contract:
      - grid is categorical piece-id ALWAYS:
          0 = empty
          1..K = kind_idx+1
      - active_kind / next_kind are STRICT kind indices 0..K-1
    """
    H, W, K = int(spec.board_h), int(spec.board_w), int(spec.num_kinds)

    grid = _extract_locked_grid(game, state)
    if grid.ndim != 2:
        raise ValueError(f"locked grid must be 2D (H,W), got shape={grid.shape}")

    if int(grid.shape[1]) != int(W):
        consists = f"got W={grid.shape[1]} expected W={W}"
        raise ValueError(f"locked grid width mismatch: {consists}")

    # If engine uses taller grid (spawn rows), take bottom H rows.
    if int(grid.shape[0]) != int(H):
        if int(grid.shape[0]) < int(H):
            raise ValueError(f"locked grid height too small: got H={grid.shape[0]} expected H={H}")
        grid = grid[-H:, :]

    g_u8 = np.asarray(grid).astype(np.uint8, copy=False)

    # Strict range check (categorical ids).
    if g_u8.size:
        mx = int(g_u8.max())
        if mx > int(K):
            raise ValueError(f"locked grid has value >K: max={mx} (K={K})")

    # Ensure dtype + clamp defensively into obs space range [0..K]
    if int(K) < 255:
        g_u8 = np.clip(g_u8, 0, int(K)).astype(np.uint8, copy=False)

    active_kind = _extract_active_kind_idx(state)
    next_kind = _extract_next_kind_idx(state)

    if not (0 <= int(active_kind) < int(K)):
        raise ValueError(f"active_kind_idx out of range: {active_kind} (K={K})")
    if not (0 <= int(next_kind) < int(K)):
        raise ValueError(f"next_kind_idx out of range: {next_kind} (K={K})")

    return {
        "grid": g_u8,
        "active_kind": int(active_kind),
        "next_kind": int(next_kind),
    }


__all__ = [
    "MacroObsSpec",
    "build_macro_obs_space",
    "encode_macro_obs",
]
