# src/tetris_rl/game/rendering/pygame/grid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pygame

from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.game.rendering.pygame.palette import Palette, Color
from tetris_rl.game.rendering.pygame.surf import SurfaceCache


# -----------------------------------------------------------------------------
# Rendering constants (no inline magic numbers)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GridRenderCfg:
    border_width: int = 2
    grid_line_width: int = 1
    hidden_line_width: int = 2
    ghost_alpha: int = 110  # 0..255
    ghost_outline_width: int = 2


CFG = GridRenderCfg()


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
        pieces: PieceSet,
        cache: SurfaceCache,
) -> None:
    """
    Draw the locked board grid, then overlay the active piece from `state.active`.
    Optionally render a "ghost" macro placement cursor (visual-only) when provided.

    STRICT CONTRACT:
      - `grid` is the LOCKED board only.
      - Board cell encoding is "board_id":
          0 = empty
          1..K = kind_idx + 1
      - No piece_id / legacy ids exist in the pipeline.
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
            pieces=pieces,
            cache=cache,
        )
        return

    ox, oy = origin
    h, w = int(arr.shape[0]), int(arr.shape[1])

    # Draw locked board cells
    for y in range(h):
        for x in range(w):
            board_id = int(arr[y, x])
            color = _board_id_to_color(board_id=board_id, palette=palette, pieces=pieces)
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

    # Optional ghost cursor (draw BEFORE active overlay so active remains readable)
    if ghost is not None:
        _draw_ghost_overlay(
            screen=screen,
            ghost=ghost,
            origin=origin,
            cell=int(cell),
            palette=palette,
            pieces=pieces,
            cache=cache,
        )

    # Overlay active piece (visual only)
    if state is not None:
        _draw_active_overlay(
            screen=screen,
            state=state,
            origin=origin,
            cell=cell,
            palette=palette,
            pieces=pieces,
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
# Overlay helpers
# -----------------------------------------------------------------------------
def _draw_active_overlay(
        *,
        screen: pygame.Surface,
        state: Any,
        origin: Tuple[int, int],
        cell: int,
        palette: Palette,
        pieces: PieceSet,
        cache: SurfaceCache,
) -> None:
    """
    Draw active piece mask at its (x,y) with its rotation.
    Does not modify board grid.
    """
    ap = getattr(state, "active", None)
    if ap is None:
        return

    kind = getattr(ap, "kind", None)
    rot = getattr(ap, "rot", None)
    px = getattr(ap, "x", None)
    py = getattr(ap, "y", None)
    if kind is None or rot is None or px is None or py is None:
        return

    kind_s = str(kind)
    try:
        m = pieces.mask(kind_s, int(rot))
    except Exception:
        return

    c = pieces.color_of(kind_s)
    color = c if c is not None else palette.fallback_piece

    ox, oy = origin
    mh, mw = int(m.shape[0]), int(m.shape[1])

    for yy in range(mh):
        for xx in range(mw):
            if int(m[yy, xx]) == 0:
                continue
            gx = int(px) + xx
            gy = int(py) + yy
            rx = ox + gx * cell
            ry = oy + gy * cell
            screen.blit(cache.cell(size=cell, color=color), (rx, ry))


def _draw_ghost_overlay(
        *,
        screen: pygame.Surface,
        ghost: dict[str, Any],
        origin: Tuple[int, int],
        cell: int,
        palette: Palette,
        pieces: PieceSet,
        cache: SurfaceCache,
) -> None:
    """
    Draw a visual-only macro placement cursor given:
      ghost["kind"] : piece kind str
      ghost["rot"]  : rotation int (asset rotation id)
      ghost["col"]  : bbox-left column int (env macro col)
      ghost["py"]   : active piece y (engine y)
      ghost["legal"]: Optional[bool] legality signal (if False -> outline warning)
    """
    kind = str(ghost.get("kind", "?"))
    rot = int(ghost.get("rot", 0))
    col = int(ghost.get("col", 0))
    py = int(ghost.get("py", 0))
    legal = ghost.get("legal", None)

    # We need to map bbox-left col to engine x. The PieceSet API doesn't expose that here,
    # so we render the ghost at "engine x = col" as a conservative visual.
    # If you want perfect alignment, expose cache.bbox_left_to_engine_x via env/render hook.
    px = int(col)

    try:
        m = pieces.mask(kind, int(rot))
    except Exception:
        return

    base = pieces.color_of(kind)
    color = base if base is not None else palette.fallback_piece

    # Draw translucent cells by blitting alpha surfaces.
    ox, oy = origin
    mh, mw = int(m.shape[0]), int(m.shape[1])

    ghost_surf = pygame.Surface((cell, cell), pygame.SRCALPHA)
    # derive RGBA from color tuple (r,g,b)
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    ghost_surf.fill((r, g, b, int(CFG.ghost_alpha)))

    for yy in range(mh):
        for xx in range(mw):
            if int(m[yy, xx]) == 0:
                continue
            gx = px + xx
            gy = py + yy
            rx = ox + gx * cell
            ry = oy + gy * cell
            screen.blit(ghost_surf, (rx, ry))

    # Outline bbox area for readability (and to show illegality)
    outline_color = palette.border if (legal is None or legal) else (240, 90, 90)
    x0 = ox + px * cell
    y0 = oy + py * cell
    pygame.draw.rect(
        screen,
        outline_color,
        pygame.Rect(x0, y0, mw * cell, mh * cell),
        width=int(CFG.ghost_outline_width),
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
        pieces: PieceSet,
        cache: SurfaceCache,
) -> None:
    ox, oy = origin
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    for y in range(h):
        row = grid[y]
        for x in range(w):
            board_id = int(row[x])
            color = _board_id_to_color(board_id=board_id, palette=palette, pieces=pieces)
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
def _board_id_to_color(*, board_id: int, palette: Palette, pieces: PieceSet) -> Color:
    """
    board_id encoding:
      0      -> empty
      1..K   -> kind_idx+1  (kind_idx = board_id-1)
    """
    if int(board_id) <= 0:
        return palette.empty

    kinds = pieces.kinds()
    idx = int(board_id) - 1
    if idx < 0 or idx >= len(kinds):
        return palette.fallback_piece

    kind = kinds[idx]
    c = pieces.color_of(kind)
    return c if c is not None else palette.fallback_piece
