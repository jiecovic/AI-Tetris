# src/tetris_rl/core/game/rendering/pygame/palette.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class Palette:
    """
    Central UI palette for pygame rendering.

    Compatibility goals:
      - Keep legacy attribute names expected by hud_panel.py / sidebar.py / grid.py
      - Add canonical piece-id coloring (1..=7) for Rust-engine grids and previews
    """

    # ---------------------------------------------------------------------
    # Core surfaces / panels
    # ---------------------------------------------------------------------
    bg: Color = (18, 18, 22)

    panel_bg: Color = (26, 26, 32)
    border: Color = (70, 70, 85)

    text: Color = (235, 235, 245)
    muted: Color = (170, 170, 190)
    matrix: Color = (0, 255, 128)

    empty: Color = (32, 32, 40)         # empty cell fill
    grid: Color = (45, 45, 58)          # grid line color
    warn: Color = (240, 90, 90)         # warnings / game over

    fallback_piece: Color = (180, 180, 180)

    # ---------------------------------------------------------------------
    # HUD panel (EXPECTED BY hud_panel.py)
    # ---------------------------------------------------------------------
    hud_bg: Color = (24, 24, 30)
    hud_border: Color = (70, 70, 85)
    hud_text: Color = (235, 235, 245)
    hud_muted: Color = (170, 170, 190)

    # ---------------------------------------------------------------------
    # Hidden rows overlay (OPTIONAL; grid.py reads these via getattr)
    # ---------------------------------------------------------------------
    hidden_rows: int = 0
    hidden_overlay_rgba: Tuple[int, int, int, int] = (0, 0, 0, 95)

    # ---------------------------------------------------------------------
    # Piece-id colors (engine encoding uses 1..=7; 0 = empty)
    #
    # Mapping matches your earlier glyph palette:
    #   I cyan, O yellow, T purple, S green, Z red, J blue, L orange
    #
    # IMPORTANT: this is UI-only.
    # ---------------------------------------------------------------------
    piece_1: Color = (0, 240, 240)    # I
    piece_2: Color = (240, 240, 0)    # O
    piece_3: Color = (160, 0, 240)    # T
    piece_4: Color = (0, 240, 0)      # S
    piece_5: Color = (240, 0, 0)      # Z
    piece_6: Color = (0, 0, 240)      # J
    piece_7: Color = (240, 160, 0)    # L

    def color_for_piece_id(self, piece_id: int) -> Color:
        """
        Map engine piece id -> RGB color.

        piece_id:
          0 => empty
          1..=7 => piece color
          otherwise => fallback_piece
        """
        pid = int(piece_id)
        if pid <= 0:
            return self.empty
        if pid == 1:
            return self.piece_1
        if pid == 2:
            return self.piece_2
        if pid == 3:
            return self.piece_3
        if pid == 4:
            return self.piece_4
        if pid == 5:
            return self.piece_5
        if pid == 6:
            return self.piece_6
        if pid == 7:
            return self.piece_7
        return self.fallback_piece
