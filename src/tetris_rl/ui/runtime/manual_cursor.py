# src/tetris_rl/ui/runtime/manual_cursor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class _CursorState:
    kind: str = "?"
    rot: int = 0
    col: int = 0  # IMPORTANT: engine action "col" parameter == bbox-left (col_left)
    y: int = 0  # render-only (spawn row), typically 0


class ManualMacroCursor:
    """
    Manual macro-placement cursor for watch mode.

    STRICT CONVENTIONS (Rust engine authoritative):
      - Cursor stores ENGINE ACTION parameters: (rot, col_left)
      - Discrete action_id is always computed via engine.encode_action_id(rot, col_left)
      - Legality is derived from engine.action_mask() at that action_id
      - Rendering uses engine-provided 4x4 preview masks (UI-only)

    Expects:
      - env.game is the PyO3 TetrisEngine
      - env.action_mode in {"discrete", "multidiscrete"}
    """

    def __init__(self, *, game: Any, env: Any) -> None:
        self.env = env
        engine = getattr(env, "game", None)
        if engine is None:
            raise RuntimeError("ManualMacroCursor expects env.game to be the PyO3 TetrisEngine")
        self.engine = engine

        self.board_w = int(self.engine.board_w())
        self.max_rots = int(self.engine.max_rots())
        self.action_dim = int(self.engine.action_dim())

        self._st = _CursorState(
            kind="?",
            rot=0,
            col=max(0, min(self.board_w // 2, self.board_w - 1)),
            y=0,
        )

        self._last_snapshot: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------
    # Snapshot sync
    # ---------------------------------------------------------------------

    def sync_from_snapshot(self, st: Dict[str, Any]) -> None:
        """
        Snapshot keys (PyO3):
          - active_kind: str
          - active_kind_idx: int in 0..6 (obs semantics)
        """
        self._last_snapshot = st
        self._st.kind = str(st.get("active_kind", "?"))

        self.board_w = int(self.engine.board_w())
        self.max_rots = int(self.engine.max_rots())
        self.action_dim = int(self.engine.action_dim())

        self._clamp()

        if not self._is_legal(rot=self._st.rot, col=self._st.col):
            rec = self._find_any_legal_near(rot=self._st.rot, col=self._st.col)
            if rec is not None:
                self._st.rot, self._st.col = rec
                self._clamp()

    def sync_from_active(self) -> None:
        st = self.engine.snapshot(include_grid=False, visible=True)
        if isinstance(st, dict):
            self.sync_from_snapshot(st)

    # ---------------------------------------------------------------------
    # UI helpers (mask-derived bbox geometry)
    # ---------------------------------------------------------------------

    def _kind_idx0(self) -> int:
        if isinstance(self._last_snapshot, dict):
            try:
                return int(self._last_snapshot.get("active_kind_idx", 0))
            except Exception:
                return 0
        return 0

    def _kind_id_1to7(self) -> int:
        return int(self._kind_idx0()) + 1

    def _preview_mask(self, *, rot: int) -> Optional[np.ndarray]:
        try:
            m = self.engine.kind_preview_mask(int(self._kind_id_1to7()), rot=int(rot))
        except Exception:
            return None
        try:
            arr = np.asarray(m)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        return arr

    @staticmethod
    def _mask_bbox_w(mask: np.ndarray) -> int:
        try:
            ys, xs = (mask != 0).nonzero()
        except Exception:
            return 0
        if xs.size == 0:
            return 0
        return int(xs.max() - xs.min() + 1)

    def _bbox_left_max_for_rot(self, rot: int) -> int:
        m = self._preview_mask(rot=int(rot))
        if m is None:
            return max(0, int(self.board_w - 1))
        bw = self._mask_bbox_w(m)
        if bw <= 0:
            return max(0, int(self.board_w - 1))
        return int(self.board_w - bw)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _clamp(self) -> None:
        self.max_rots = int(self.engine.max_rots())
        self.board_w = int(self.engine.board_w())

        self._st.rot = int(max(0, min(int(self._st.rot), int(self.max_rots - 1))))

        col_max = self._bbox_left_max_for_rot(int(self._st.rot))
        self._st.col = int(max(0, min(int(self._st.col), int(col_max))))

        self._st.y = int(max(0, int(self._st.y)))

    def _mask_bool(self) -> np.ndarray:
        m_u8 = np.asarray(self.engine.action_mask(), dtype=np.uint8).reshape(-1)
        return m_u8 != 0

    def _action_id(self, *, rot: int, col: int) -> int:
        return int(self.engine.encode_action_id(int(rot), int(col)))

    def _is_legal(self, *, rot: int, col: int) -> bool:
        try:
            aid = self._action_id(rot=int(rot), col=int(col))
        except Exception:
            return False
        mask = self._mask_bool()
        return bool(0 <= aid < int(mask.size) and bool(mask[aid]))

    def _find_any_legal_near(self, *, rot: int, col: int) -> Optional[tuple[int, int]]:
        self._clamp()
        rot = int(max(0, min(int(rot), int(self.max_rots - 1))))

        col_max = self._bbox_left_max_for_rot(rot)
        col = int(max(0, min(int(col), int(col_max))))

        for d in range(0, int(col_max) + 1):
            for c in (int(col) - d, int(col) + d):
                if 0 <= c <= int(col_max) and self._is_legal(rot=int(rot), col=int(c)):
                    return (int(rot), int(c))

        for r in range(0, int(self.max_rots)):
            cm = self._bbox_left_max_for_rot(int(r))
            for c in range(0, int(cm) + 1):
                if self._is_legal(rot=int(r), col=int(c)):
                    return (int(r), int(c))

        return None

    def _piece_glyph(self) -> str:
        k = str(self._st.kind).strip().upper()
        return k[:1] if k else "?"

    def _kick_offsets_for_piece(self) -> list[int]:
        g = self._piece_glyph()
        if g == "I":
            return [0, -2, +2, -1, +1]
        if g == "O":
            return [0]
        return [0, -1, +1, -2, +2]

    # ---------------------------------------------------------------------
    # Cursor movement
    # ---------------------------------------------------------------------

    def move_x(self, dx: int) -> None:
        self._clamp()
        step = 1 if int(dx) >= 0 else -1

        if not self._is_legal(rot=int(self._st.rot), col=int(self._st.col)):
            rec = self._find_any_legal_near(rot=int(self._st.rot), col=int(self._st.col))
            if rec is not None:
                self._st.rot, self._st.col = rec
                self._clamp()

        col_max = self._bbox_left_max_for_rot(int(self._st.rot))
        c = int(self._st.col)

        for _ in range(int(col_max) + 2):
            c += step
            if c < 0 or c > int(col_max):
                break
            if self._is_legal(rot=int(self._st.rot), col=int(c)):
                self._st.col = int(c)
                self._clamp()
                return

    def move_col(self, dx: int) -> None:
        self.move_x(dx)

    def move_rot(self, dr: int) -> None:
        self._clamp()
        step = 1 if int(dr) >= 0 else -1

        if not self._is_legal(rot=int(self._st.rot), col=int(self._st.col)):
            rec = self._find_any_legal_near(rot=int(self._st.rot), col=int(self._st.col))
            if rec is not None:
                self._st.rot, self._st.col = rec
                self._clamp()

        r0 = int(self._st.rot)
        c0 = int(self._st.col)
        kicks = self._kick_offsets_for_piece()

        r = r0
        for _ in range(int(self.max_rots)):
            r = (r + step) % int(self.max_rots)
            col_max = self._bbox_left_max_for_rot(int(r))

            for k in kicks:
                c = int(c0) + int(k)
                if c < 0 or c > int(col_max):
                    continue
                if self._is_legal(rot=int(r), col=int(c)):
                    self._st.rot = int(r)
                    self._st.col = int(c)
                    self._clamp()
                    return

    # ---------------------------------------------------------------------
    # Pause behavior: ALWAYS recenter
    # ---------------------------------------------------------------------

    def recenter_for_pause(self) -> None:
        """
        Called when pause toggles ON.

        Always center the piece (bbox-left) for the CURRENT rotation, ignoring any previous cursor state.
        """
        self._clamp()
        m = self._preview_mask(rot=int(self._st.rot))
        if m is None:
            self._st.col = max(0, (int(self.board_w) - 1) // 2)
            self._clamp()
            return

        bw = self._mask_bbox_w(m)
        if bw <= 0:
            self._st.col = max(0, (int(self.board_w) - 1) // 2)
            self._clamp()
            return

        col_left = int((int(self.board_w) - int(bw)) // 2)
        col_max = self._bbox_left_max_for_rot(int(self._st.rot))
        self._st.col = int(max(0, min(int(col_left), int(col_max))))
        self._clamp()

    # ---------------------------------------------------------------------
    # Outputs for watch loop / renderer
    # ---------------------------------------------------------------------

    def ghost_for_render(self, enabled: bool) -> Optional[dict[str, Any]]:
        if not bool(enabled):
            return None

        self._clamp()
        legal = self._is_legal(rot=int(self._st.rot), col=int(self._st.col))

        m = self._preview_mask(rot=int(self._st.rot))
        if m is None:
            return None

        return {
            "mask": m,
            "kind_idx": int(self._kind_id_1to7()),
            "rot": int(self._st.rot),
            "col": int(self._st.col),
            "x": int(self._st.col),  # legacy key (still bbox-left)
            "y": int(self._st.y),
            "legal": bool(legal),
        }

    def action_for_commit(self) -> Any:
        self._clamp()
        action_mode = str(getattr(self.env, "action_mode", "discrete")).strip().lower()
        if action_mode == "discrete":
            return int(self._action_id(rot=int(self._st.rot), col=int(self._st.col)))
        return (int(self._st.rot), int(self._st.col))
