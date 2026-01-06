# src/tetris_rl/game/rendering/pygame/sidebar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import pygame

from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.game.rendering.pygame.palette import Palette, Color
from tetris_rl.game.rendering.pygame.surf import SurfaceCache, blit_text

# -----------------------------------------------------------------------------
# Public sizing contract
# -----------------------------------------------------------------------------
# Single source of truth for how much horizontal space the sidebar needs.
# The window/layout code should size the window using this value.
SIDEBAR_W = 280


# -----------------------------------------------------------------------------
# Layout constants (all magic numbers live here, not inline)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SidebarLayout:
    """
    Centralizes pixel geometry for the sidebar panels.

    Keep all "magic numbers" here so render code stays readable and tweakable.
    """
    # Vertical spacing between panels
    panel_gap_y: int = 14

    # Heights of panels (px)
    next_panel_h: int = 170
    stats_panel_h: int = 230
    controls_min_h: int = 96

    # Panel title padding
    title_pad_x: int = 10
    title_pad_y: int = 8

    # NEXT: label placement
    next_label_y_offset: int = 32

    # NEXT: preview box placement
    next_box_y_offset: int = 52
    next_box_border_w: int = 2

    # STATS: internal padding and row layout
    stats_pad_x: int = 10
    stats_header_y_offset: int = 34
    stats_header_row_gap_y: int = 22
    stats_row_h: int = 20

    # STATS: two-column geometry (value x is label x + value_dx)
    stats_value_dx: int = 68

    # CONTROLS: list layout
    controls_list_y_offset: int = 36
    controls_key_x_offset: int = 10
    controls_desc_x_offset: int = 90
    controls_row_h: int = 20

    # CONTROLS: footer placement
    controls_footer_needed_h: int = 120
    controls_game_over_y_from_bottom: int = 34
    controls_hint_y_from_bottom: int = 16


_LAYOUT = SidebarLayout()


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _fmt_value(v: Any) -> str:
    """
    Generic "safe display" formatter for values coming from env/game.
    Sidebar does not interpret semantics beyond simple display.
    """
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    try:
        if isinstance(v, int):
            return f"{int(v)}"
        if isinstance(v, float):
            return f"{float(v):.2f}"

        import numpy as np

        if isinstance(v, np.generic):
            return _fmt_value(v.item())
    except Exception:
        pass
    return str(v)


def _extract_env_rows(env_info: Optional[dict[str, Any]]) -> List[tuple[str, Any]]:
    """
    The env is responsible for deciding what "ENV" rows exist.
    Contract:
      env_info may contain:
        - env_info["sidebar_env"] = [("Key", value), ...]  (list/tuple of 2-tuples)
        - OR env_info["sidebar_env"] = {"Key": value, ...} (dict)
    Sidebar will render whatever is present, otherwise render nothing in ENV column.
    """
    if not isinstance(env_info, dict):
        return []

    rows = env_info.get("sidebar_env", None)
    if rows is None:
        return []

    if isinstance(rows, dict):
        return [(str(k), rows[k]) for k in rows.keys()]

    if isinstance(rows, (list, tuple)):
        out: List[tuple[str, Any]] = []
        for item in rows:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), item[1]))
        return out

    return []


# -----------------------------------------------------------------------------
# Public draw
# -----------------------------------------------------------------------------
def draw_sidebar(
        *,
        screen: pygame.Surface,
        state: Any,
        env_info: Optional[dict[str, Any]] = None,
        game_metrics: Optional[dict[str, Any]] = None,
        reward: float,
        done: bool,
        x: int,
        y: int,
        w: int,
        origin: Tuple[int, int],
        margin: int,
        grid: Any,
        cell: int,
        palette: Palette,
        pieces: PieceSet,
        cache: SurfaceCache,
        font_small: pygame.font.Font,
        font_tiny: pygame.font.Font,
) -> None:
    """
    Render the right sidebar. This file is strictly presentation:
      - GAME column uses game_metrics provided by renderer (authoritative)
      - ENV column uses env_info["sidebar_env"] provided by env (optional)
    """
    panel_w = int(w)

    # Determine available height to size the CONTROLS panel.
    grid_h = len(grid)
    board_outer_h = int(grid_h) * int(cell) + 2 * int(margin)

    used_h = _LAYOUT.next_panel_h + _LAYOUT.panel_gap_y + _LAYOUT.stats_panel_h + _LAYOUT.panel_gap_y
    controls_h = max(_LAYOUT.controls_min_h, int(board_outer_h) - int(used_h))

    # -------------------------------------------------------------------------
    # NEXT panel
    # -------------------------------------------------------------------------
    _panel(
        screen=screen,
        palette=palette,
        font_small=font_small,
        x=x,
        y=y,
        w=panel_w,
        h=_LAYOUT.next_panel_h,
        title="NEXT",
    )

    next_kind = getattr(state, "next_kind", None)
    next_label = f"{next_kind if next_kind else 'n/a'}"

    label_w = font_tiny.size(next_label)[0]
    label_x = int(x) + (int(panel_w) - int(label_w)) // 2
    label_y = int(y) + int(_LAYOUT.next_label_y_offset)

    blit_text(screen=screen, font=font_tiny, text=next_label, pos=(label_x, label_y), color=palette.muted)

    next_box_cells = 4
    box_w = int(next_box_cells) * int(cell)
    box_h = int(next_box_cells) * int(cell)

    box_x = int(x) + (int(panel_w) - int(box_w)) // 2
    box_y = int(y) + int(_LAYOUT.next_box_y_offset)

    box_rect = pygame.Rect(int(box_x), int(box_y), int(box_w), int(box_h))
    pygame.draw.rect(screen, palette.empty, box_rect)
    pygame.draw.rect(screen, palette.border, box_rect, width=int(_LAYOUT.next_box_border_w))

    if next_kind and next_kind in pieces:
        draw_piece_preview(
            screen=screen,
            kind=str(next_kind),
            dst_x=int(box_x),
            dst_y=int(box_y),
            cells=int(next_box_cells),
            cell=int(cell),
            palette=palette,
            pieces=pieces,
            cache=cache,
        )

    # -------------------------------------------------------------------------
    # STATS panel
    # -------------------------------------------------------------------------
    stats_y = int(y) + int(_LAYOUT.next_panel_h) + int(_LAYOUT.panel_gap_y)
    _panel(
        screen=screen,
        palette=palette,
        font_small=font_small,
        x=x,
        y=stats_y,
        w=panel_w,
        h=_LAYOUT.stats_panel_h,
        title="STATS",
    )

    score = _as_int(getattr(state, "score", 0), 0)
    lines = _as_int(getattr(state, "lines", 0), 0)
    level = _as_int(getattr(state, "level", 0), 0)

    # "done" is renderer-supplied; state may have its own game_over/done too.
    game_over_val = getattr(state, "game_over", None)
    if game_over_val is None:
        game_over_val = getattr(state, "done", None)
    game_over = bool(done) if game_over_val is None else bool(game_over_val)

    gm = game_metrics if isinstance(game_metrics, dict) else {}
    holes = gm.get("holes", None)
    max_height = gm.get("max_height", None)

    # Two columns inside STATS
    col1_label_x = int(x) + int(_LAYOUT.stats_pad_x)
    col1_value_x = int(col1_label_x) + int(_LAYOUT.stats_value_dx)

    col2_label_x = int(x) + (int(panel_w) // 2) + int(_LAYOUT.stats_pad_x)
    col2_value_x = int(col2_label_x) + int(_LAYOUT.stats_value_dx)

    yy = int(stats_y) + int(_LAYOUT.stats_header_y_offset)

    blit_text(screen=screen, font=font_tiny, text="GAME", pos=(col1_label_x, yy), color=palette.matrix)
    blit_text(screen=screen, font=font_tiny, text="ENV", pos=(col2_label_x, yy), color=palette.matrix)
    yy += int(_LAYOUT.stats_header_row_gap_y)

    game_rows: List[tuple[str, Any]] = [
        ("Score", score),
        ("Lines", lines),
        ("Level", level),
        ("Holes", holes),
        ("MaxH", max_height),
        ("Over", bool(game_over)),
    ]
    env_rows = _extract_env_rows(env_info)

    n_rows = max(len(game_rows), len(env_rows), 1)
    for i in range(n_rows):
        if i < len(game_rows):
            k, v = game_rows[i]
            _draw_kv(
                screen=screen,
                font=font_tiny,
                palette=palette,
                k=str(k),
                v=v,
                label_x=col1_label_x,
                value_x=col1_value_x,
                y=yy,
            )

        if i < len(env_rows):
            k, v = env_rows[i]
            _draw_kv(
                screen=screen,
                font=font_tiny,
                palette=palette,
                k=str(k),
                v=v,
                label_x=col2_label_x,
                value_x=col2_value_x,
                y=yy,
            )

        yy += int(_LAYOUT.stats_row_h)

    # -------------------------------------------------------------------------
    # CONTROLS panel
    # -------------------------------------------------------------------------
    ctrl_y = int(stats_y) + int(_LAYOUT.stats_panel_h) + int(_LAYOUT.panel_gap_y)
    _panel(
        screen=screen,
        palette=palette,
        font_small=font_small,
        x=x,
        y=ctrl_y,
        w=panel_w,
        h=controls_h,
        title="CONTROLS",
    )

    legend = [
        ("A / D", "move left/right"),
        ("S", "soft drop"),
        ("W / Space", "hard drop"),
        ("Q / E", "rotate CCW/CW"),
        ("R", "reset"),
        ("Esc", "quit"),
    ]

    yy = int(ctrl_y) + int(_LAYOUT.controls_list_y_offset)
    key_x = int(x) + int(_LAYOUT.controls_key_x_offset)
    desc_x = int(x) + int(_LAYOUT.controls_desc_x_offset)

    # Don't let list overwrite panel bottom (reserve a little breathing room).
    bottom_guard = int(ctrl_y) + int(controls_h) - int(_LAYOUT.controls_row_h) - int(_LAYOUT.title_pad_y)

    for key, desc in legend:
        if yy > bottom_guard:
            break
        blit_text(screen=screen, font=font_tiny, text=f"{key:<9}", pos=(key_x, yy), color=palette.matrix)
        blit_text(screen=screen, font=font_tiny, text=desc, pos=(desc_x, yy), color=palette.muted)
        yy += int(_LAYOUT.controls_row_h)

    if bool(game_over) and int(controls_h) >= int(_LAYOUT.controls_footer_needed_h):
        go_y = int(ctrl_y) + int(controls_h) - int(_LAYOUT.controls_game_over_y_from_bottom)
        hint_y = int(ctrl_y) + int(controls_h) - int(_LAYOUT.controls_hint_y_from_bottom)

        blit_text(screen=screen, font=font_small, text="GAME OVER", pos=(key_x, go_y), color=palette.warn)
        blit_text(screen=screen, font=font_tiny, text="Press R to reset", pos=(key_x, hint_y), color=palette.muted)


# -----------------------------------------------------------------------------
# Small primitives
# -----------------------------------------------------------------------------
def _draw_kv(
        *,
        screen: pygame.Surface,
        font: pygame.font.Font,
        palette: Palette,
        k: str,
        v: Any,
        label_x: int,
        value_x: int,
        y: int,
) -> None:
    label = f"{k:>5}:"
    blit_text(screen=screen, font=font, text=label, pos=(int(label_x), int(y)), color=palette.muted)
    blit_text(screen=screen, font=font, text=_fmt_value(v), pos=(int(value_x), int(y)), color=palette.text)


def draw_piece_preview(
        *,
        screen: pygame.Surface,
        kind: str,
        dst_x: int,
        dst_y: int,
        cells: int,
        cell: int,
        palette: Palette,
        pieces: PieceSet,
        cache: SurfaceCache,
) -> None:
    m = pieces.mask(kind, 0)

    # Color comes from PieceSet by kind (no id indirection).
    c = pieces.color_of(kind)
    color: Color = c if c is not None else palette.fallback_piece

    ys, xs = (m != 0).nonzero()
    if len(xs) == 0:
        return

    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())
    shape_w = maxx - minx + 1
    shape_h = maxy - miny + 1

    off_x = (int(cells) - int(shape_w)) // 2
    off_y = (int(cells) - int(shape_h)) // 2

    for yy in range(int(shape_h)):
        for xx in range(int(shape_w)):
            if m[int(miny) + yy, int(minx) + xx] == 0:
                continue
            rx = int(dst_x) + (int(off_x) + int(xx)) * int(cell)
            ry = int(dst_y) + (int(off_y) + int(yy)) * int(cell)
            screen.blit(cache.cell(size=int(cell), color=color), (int(rx), int(ry)))


def _panel(
        *,
        screen: pygame.Surface,
        palette: Palette,
        font_small: pygame.font.Font,
        x: int,
        y: int,
        w: int,
        h: int,
        title: Optional[str] = None,
) -> None:
    rect = pygame.Rect(int(x), int(y), int(w), int(h))
    pygame.draw.rect(screen, palette.panel_bg, rect)
    pygame.draw.rect(screen, palette.border, rect, width=2)
    if title:
        tx = int(x) + int(_LAYOUT.title_pad_x)
        ty = int(y) + int(_LAYOUT.title_pad_y)
        blit_text(screen=screen, font=font_small, text=title, pos=(tx, ty), color=palette.matrix)
