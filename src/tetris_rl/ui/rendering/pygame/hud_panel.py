# src/tetris_rl/ui/rendering/pygame/hud_panel.py
from __future__ import annotations

from dataclasses import dataclass

import pygame

from tetris_rl.ui.rendering.pygame.palette import Palette
from tetris_rl.ui.rendering.pygame.surf import blit_text


@dataclass(frozen=True)
class HudFonts:
    header: pygame.font.Font
    body: pygame.font.Font


def hud_panel_height_for_lines(*, fonts: HudFonts, n_lines: int, pad_y: int = 10, gap_y: int = 3) -> int:
    """
    Simple sizing helper for a HUD that renders n_lines (including blanks).
    We treat all lines as body height for sizing; header lines are slightly taller,
    so callers can add a small buffer if desired.
    """
    n = max(1, int(n_lines))
    h = pad_y
    h += n * (fonts.body.get_height() + gap_y)
    h -= gap_y
    h += pad_y
    return int(h + 2)


def draw_hud_panel(*, screen: pygame.Surface, palette: Palette, fonts: HudFonts, hud_height: int, text: str) -> None:
    """
    Render a structured HUD text into a top panel.

    Conventions expected from ui/runtime/hud_text.py:
      - Section headers start with "# " (e.g. "# Status")
      - Key/value lines look like "Label: Value"
      - Optional: multiple key/value segments on one line separated by " | "
        Example: "Score: 10 | Lines: 2"
        -> rendered as two columns on the same row.
      - Blank lines are allowed
    """
    w = int(screen.get_width())
    h = int(hud_height)

    rect = pygame.Rect(0, 0, w, h)
    pygame.draw.rect(screen, palette.hud_bg, rect)
    pygame.draw.rect(screen, palette.hud_border, rect, width=1)

    pad_x = 14
    pad_y = 10
    top_extra = 2  # avoids occasional ascent clipping on Windows fonts
    gap_y = 3
    max_px = w - 2 * pad_x

    raw = text.splitlines()
    while raw and raw[-1].strip() == "":
        raw.pop()
    if not raw:
        return

    # 2-column layout geometry
    col_gap = 26
    col_w = max(0, (w - 2 * pad_x - col_gap) // 2)
    col_xs = [pad_x, pad_x + col_w + col_gap]

    y = pad_y + top_extra

    for ln in raw:
        if y + fonts.body.get_height() > h - pad_y:
            break

        if ln.strip() == "":
            y += max(8, fonts.body.get_height() // 2)
            continue

        s = ln.rstrip("\n")

        # Header: "# Title"
        if s.lstrip().startswith("# "):
            title = s.lstrip()[2:].strip()
            title = title[:200] if title else ""
            title = _fit_text(fonts.header, title, max_px=max_px)
            if title:
                blit_text(screen=screen, font=fonts.header, text=title, pos=(pad_x, y), color=palette.matrix)
            y += fonts.header.get_height() + gap_y
            continue

        # Key/value row: supports " | " segments for 2 columns
        if ":" in s:
            segments = [seg.strip() for seg in s.split(" | ") if seg.strip()]

            if len(segments) == 1:
                _draw_pair(
                    screen=screen,
                    palette=palette,
                    font=fonts.body,
                    x0=pad_x,
                    y=y,
                    max_width=max_px,
                    pair_text=segments[0],
                )
            else:
                _draw_pair(
                    screen=screen,
                    palette=palette,
                    font=fonts.body,
                    x0=col_xs[0],
                    y=y,
                    max_width=col_w,
                    pair_text=segments[0],
                )
                _draw_pair(
                    screen=screen,
                    palette=palette,
                    font=fonts.body,
                    x0=col_xs[1],
                    y=y,
                    max_width=col_w,
                    pair_text=segments[1],
                )

            y += fonts.body.get_height() + gap_y
            continue

        # Fallback line (rare)
        line = _fit_text(fonts.body, s.strip(), max_px=max_px)
        if line:
            blit_text(screen=screen, font=fonts.body, text=line, pos=(pad_x, y), color=palette.muted)
            y += fonts.body.get_height() + gap_y


def _draw_pair(
        *,
        screen: pygame.Surface,
        palette: Palette,
        font: pygame.font.Font,
        x0: int,
        y: int,
        max_width: int,
        pair_text: str,
) -> None:
    """
    Render one "Label: Value" pair within a bounded width starting at x0.
    """
    if ":" not in pair_text:
        line = _fit_text(font, pair_text, max_px=max_width)
        if line:
            blit_text(screen=screen, font=font, text=line, pos=(x0, y), color=palette.muted)
        return

    left, right = pair_text.split(":", 1)
    label = (left.strip() + ":").strip()
    value = right.strip()

    # Color values sparingly
    value_color = palette.text
    up = (label + " " + value).upper()
    if ("PAUSED" in up or "PAUSE" in up) and "YES" in up:
        value_color = palette.warn
    if "GAME OVER" in up:
        value_color = palette.warn

    # Per-pair label width (avoid labels eating all space)
    lbl_max = min(int(max_width * 0.55), max_width)
    lbl_fit = _fit_text(font, label, max_px=lbl_max)

    lbl_px = min(font.size(lbl_fit)[0] if lbl_fit else 0, lbl_max)
    val_x = x0 + lbl_px + 10

    if lbl_fit:
        blit_text(screen=screen, font=font, text=lbl_fit, pos=(x0, y), color=palette.muted)

    val_max = max(0, (x0 + max_width) - val_x)
    val_fit = _fit_text(font, value, max_px=val_max)
    if val_fit:
        blit_text(screen=screen, font=font, text=val_fit, pos=(val_x, y), color=value_color)


def _fit_text(font: pygame.font.Font, text: str, *, max_px: int) -> str:
    if font.size(text)[0] <= max_px:
        return text
    ell = "…"
    if font.size(ell)[0] > max_px:
        return ""
    s = text
    while s and font.size(s + ell)[0] > max_px:
        s = s[:-1]
    return s + ell
