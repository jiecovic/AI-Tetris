# src/tetris_rl/ui/rendering/pygame/grid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pygame

from tetris_rl.ui.rendering.pygame.palette import Palette, Color
from tetris_rl.ui.rendering.pygame.surf import SurfaceCache


# -----------------------------------------------------------------------------
# Rendering constants (no inline magic numbers)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GridRenderCfg:
    border_width: int = 2
    grid_line_width: int = 1
    hidden_line_width: int = 2
    ghost_alpha: int = 110  # 0..255

    # UI-only "active piece preview" placement in spawn rows
    active_spawn_y: int = 0


CFG = GridRenderCfg()


# -----------------------------------------------------------------------------
# Small access helpers
# -----------------------------------------------------------------------------
def _get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _try_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _try_engine_preview_mask(*, engine: Any | None, kind_id: int | None, rot: int = 0) -> Any | None:
    """
    UI-only: ask Rust engine (PyO3) for a 4x4 preview mask.

    Expected API:
      engine.kind_preview_mask(kind_id: int(1..7), rot: int=0) -> numpy uint8[4,4]
    """
    if engine is None or kind_id is None:
        return None
    try:
        ki = int(kind_id)
    except Exception:
        return None
    if ki < 1 or ki > 7:
        return None
    try:
        return engine.kind_preview_mask(ki, rot=int(rot))
    except Exception:
        return None


def _mask_nonzero_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Return (minx, miny, maxx, maxy) of nonzero cells in mask, or None if empty.
    """
    try:
        ys, xs = (mask != 0).nonzero()
    except Exception:
        return None
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_min_dx(mask: np.ndarray) -> int:
    """
    Engine-equivalent min_dx for bbox_left_to_anchor_x:
      min_dx == min x-index among filled cells in the 4x4 mask.
    """
    bbox = _mask_nonzero_bbox(mask)
    if bbox is None:
        return 0
    minx, _miny, _maxx, _maxy = bbox
    return int(minx)


def _mask_bbox_w(mask: np.ndarray) -> int:
    bbox = _mask_nonzero_bbox(mask)
    if bbox is None:
        return 0
    minx, _miny, maxx, _maxy = bbox
    return int(maxx - minx + 1)


def _draw_mask_cells_anchor_space(
    *,
    screen: pygame.Surface,
    mask: np.ndarray,
    origin: Tuple[int, int],
    board_w: int,
    board_h: int,
    cell: int,
    palette: Palette,
    cache: SurfaceCache,
    anchor_x: int,
    anchor_y: int,
    alpha: Optional[int] = None,
) -> None:
    """
    Draw mask cells at board coords (anchor_x + xx, anchor_y + yy) for each nonzero (yy,xx).

    NOTE: This treats mask indices as (dx,dy) offsets in engine's rotations() space.
    """
    ox, oy = origin

    ghost_surf_cache: dict[int, pygame.Surface] = {}

    for yy in range(int(mask.shape[0])):
        for xx in range(int(mask.shape[1])):
            try:
                v = int(mask[yy, xx])
            except Exception:
                continue
            if v <= 0:
                continue

            gx = int(anchor_x) + int(xx)
            gy = int(anchor_y) + int(yy)

            # Clip to board
            if gx < 0 or gx >= int(board_w) or gy < 0 or gy >= int(board_h):
                continue

            color: Color = palette.color_for_piece_id(v)
            rx = ox + gx * cell
            ry = oy + gy * cell

            if alpha is None:
                screen.blit(cache.cell(size=cell, color=color), (rx, ry))
            else:
                key = int(v)
                surf = ghost_surf_cache.get(key)
                if surf is None:
                    surf = pygame.Surface((cell, cell), pygame.SRCALPHA)
                    r, g, b = int(color[0]), int(color[1]), int(color[2])
                    surf.fill((r, g, b, int(alpha)))
                    ghost_surf_cache[key] = surf
                screen.blit(surf, (rx, ry))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def draw_grid(
    *,
    screen: pygame.Surface,
    grid: Any,
    state: Optional[Any] = None,
    ghost: Optional[dict[str, Any]] = None,
    origin: Tuple[int, int],
    margin: int,
    cell: int,
    show_grid_lines: bool,
    palette: Palette,
    cache: SurfaceCache,
    engine: Any = None,  # Rust engine (PyO3), UI-only preview masks
    show_active: bool = True,  # renderer-controlled
) -> None:
    """
    Draw the locked board grid. Optionally render:
      - active piece preview (spawn rows, UI-only) when show_active=True
      - ghost macro cursor (paused-only) when ghost != None

    IMPORTANT:
      - Engine action col is bbox-left (col_left), NOT anchor-x.
      - We render consistently with engine by converting:
            anchor_x = col_left - min_dx(mask)
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        _draw_grid_fallback(
            screen=screen,
            grid=grid,
            origin=origin,
            margin=margin,
            cell=cell,
            show_grid_lines=show_grid_lines,
            palette=palette,
            cache=cache,
        )
        return

    ox, oy = origin
    h, w = int(arr.shape[0]), int(arr.shape[1])

    # Draw locked board cells
    for y in range(h):
        for x in range(w):
            piece_id = int(arr[y, x])
            color = _piece_id_to_color(piece_id=piece_id, palette=palette)
            rx = ox + x * cell
            ry = oy + y * cell
            screen.blit(cache.cell(size=cell, color=color), (rx, ry))
            if show_grid_lines:
                pygame.draw.rect(
                    screen,
                    palette.grid,
                    pygame.Rect(rx, ry, cell, cell),
                    width=int(CFG.grid_line_width),
                )

    # Active piece preview in spawn rows (unpaused)
    if bool(show_active) and state is not None and engine is not None:
        _draw_active_preview(
            screen=screen,
            state=state,
            origin=origin,
            board_w=w,
            board_h=h,
            cell=int(cell),
            palette=palette,
            cache=cache,
            engine=engine,
        )

    # Ghost cursor (paused-only; no border/box)
    if ghost is not None:
        _draw_ghost_overlay(
            screen=screen,
            ghost=ghost,
            origin=origin,
            board_w=w,
            board_h=h,
            cell=int(cell),
            palette=palette,
            cache=cache,
        )

    # Outer border
    pygame.draw.rect(
        screen,
        palette.border,
        pygame.Rect(ox - margin, oy - margin, w * cell + 2 * margin, h * cell + 2 * margin),
        width=int(CFG.border_width),
    )

    # Hidden rows overlay (top area)
    hidden_rows = int(getattr(palette, "hidden_rows", 0))
    if hidden_rows > 0:
        overlay_h = min(hidden_rows, h) * cell
        if overlay_h > 0:
            overlay = pygame.Surface((w * cell, overlay_h), pygame.SRCALPHA)
            overlay.fill(palette.hidden_overlay_rgba)
            screen.blit(overlay, (ox, oy))
            pygame.draw.line(
                screen,
                palette.border,
                (ox, oy + overlay_h),
                (ox + w * cell, oy + overlay_h),
                int(CFG.hidden_line_width),
            )


# -----------------------------------------------------------------------------
# Active preview (UI-only)
# -----------------------------------------------------------------------------
def _draw_active_preview(
    *,
    screen: pygame.Surface,
    state: Any,
    origin: Tuple[int, int],
    board_w: int,
    board_h: int,
    cell: int,
    palette: Palette,
    cache: SurfaceCache,
    engine: Any,
) -> None:
    """
    Render a 4x4 preview mask for the ACTIVE piece in the spawn rows.

    Snapshot provides active_kind_idx in 0..6 (obs semantics), so map -> 1..7 for engine mask API.

    Placement matches engine geometry:
      - Choose centered bbox-left:
            col_left = (W - bbox_w)//2
      - Convert to anchor-x:
            anchor_x = col_left - min_dx(mask)
    """
    kind_idx0 = _try_int(_get(state, "active_kind_idx", None))
    if kind_idx0 is None:
        return

    kind_id = int(kind_idx0) + 1
    mask0 = _try_engine_preview_mask(engine=engine, kind_id=kind_id, rot=0)
    if mask0 is None:
        return

    try:
        m = np.asarray(mask0)
    except Exception:
        return
    if m.ndim != 2:
        return

    bbox_w = _mask_bbox_w(m)
    if bbox_w <= 0:
        return

    col_left = int((int(board_w) - int(bbox_w)) // 2)
    min_dx = _mask_min_dx(m)
    anchor_x = int(col_left) - int(min_dx)

    anchor_y = int(CFG.active_spawn_y)

    _draw_mask_cells_anchor_space(
        screen=screen,
        mask=m,
        origin=origin,
        board_w=int(board_w),
        board_h=int(board_h),
        cell=int(cell),
        palette=palette,
        cache=cache,
        anchor_x=int(anchor_x),
        anchor_y=int(anchor_y),
        alpha=None,
    )


# -----------------------------------------------------------------------------
# Ghost overlay (UI-only; no outline)
# -----------------------------------------------------------------------------
def _draw_ghost_overlay(
    *,
    screen: pygame.Surface,
    ghost: dict[str, Any],
    origin: Tuple[int, int],
    board_w: int,
    board_h: int,
    cell: int,
    palette: Palette,
    cache: SurfaceCache,
) -> None:
    """
    Draw a visual-only macro placement cursor using an engine-provided 4x4 mask.

    IMPORTANT:
      ghost["col"] (or legacy ghost["x"]) is bbox-left (engine action col_left).
      We convert to anchor-x for drawing:
        anchor_x = col_left - min_dx(mask)

    NO OUTLINE BOX is drawn (as requested).
    """
    mask = ghost.get("mask", None)
    if mask is None:
        return

    try:
        m = np.asarray(mask)
    except Exception:
        return
    if m.ndim != 2:
        return

    col_left = ghost.get("col", ghost.get("x", 0))
    try:
        col_left = int(col_left)
    except Exception:
        col_left = 0

    py = ghost.get("y", 0)
    try:
        py = int(py)
    except Exception:
        py = 0

    min_dx = _mask_min_dx(m)
    anchor_x = int(col_left) - int(min_dx)
    anchor_y = int(py)

    _draw_mask_cells_anchor_space(
        screen=screen,
        mask=m,
        origin=origin,
        board_w=int(board_w),
        board_h=int(board_h),
        cell=int(cell),
        palette=palette,
        cache=cache,
        anchor_x=int(anchor_x),
        anchor_y=int(anchor_y),
        alpha=int(CFG.ghost_alpha),
    )


# -----------------------------------------------------------------------------
# Fallback path for weird grid inputs
# -----------------------------------------------------------------------------
def _draw_grid_fallback(
    *,
    screen: pygame.Surface,
    grid: Any,
    origin: Tuple[int, int],
    margin: int,
    cell: int,
    show_grid_lines: bool,
    palette: Palette,
    cache: SurfaceCache,
) -> None:
    ox, oy = origin
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    for y in range(h):
        row = grid[y]
        for x in range(w):
            piece_id = int(row[x])
            color = _piece_id_to_color(piece_id=piece_id, palette=palette)
            rx = ox + x * cell
            ry = oy + y * cell
            screen.blit(cache.cell(size=cell, color=color), (rx, ry))
            if show_grid_lines:
                pygame.draw.rect(
                    screen,
                    palette.grid,
                    pygame.Rect(rx, ry, cell, cell),
                    width=int(CFG.grid_line_width),
                )

    pygame.draw.rect(
        screen,
        palette.border,
        pygame.Rect(ox - margin, oy - margin, w * cell + 2 * margin, h * cell + 2 * margin),
        width=int(CFG.border_width),
    )

    hidden_rows = int(getattr(palette, "hidden_rows", 0))
    if hidden_rows > 0:
        overlay_h = min(hidden_rows, h) * cell
        if overlay_h > 0:
            overlay = pygame.Surface((w * cell, overlay_h), pygame.SRCALPHA)
            overlay.fill(palette.hidden_overlay_rgba)
            screen.blit(overlay, (ox, oy))
            pygame.draw.line(
                screen,
                palette.border,
                (ox, oy + overlay_h),
                (ox + w * cell, oy + overlay_h),
                int(CFG.hidden_line_width),
            )


# -----------------------------------------------------------------------------
# Color mapping
# -----------------------------------------------------------------------------
def _piece_id_to_color(*, piece_id: int, palette: Palette) -> Color:
    if int(piece_id) <= 0:
        return palette.empty
    return palette.color_for_piece_id(int(piece_id))
