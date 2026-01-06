# src/tetris_rl/runs/manual_cursor.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from tetris_rl.game.core.macro_legality import macro_illegal_reason_bbox_left
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.game.core.types import ActivePiece, State


class ManualMacroCursor:
    def __init__(self, *, game: Any, env: Any) -> None:
        self.game = game
        self.env = env
        self.board_w = int(getattr(game, "w", 10))
        self.max_rots = int(getattr(env, "max_rots", 4))

        legal_cache = getattr(env, "_legal_cache", None)
        if legal_cache is None:
            legal_cache = StaticPlacementCache.build(pieces=game.pieces, board_w=int(self.board_w))
        self.legal_cache = legal_cache

        self.ghost_rot = 0
        self.ghost_col = max(0, min(int(self.board_w // 2), int(self.board_w - 1)))

    def sync_from_active(self) -> None:
        st = self.game.state() if hasattr(self.game, "state") else self.game._state()  # type: ignore[attr-defined]
        ap = getattr(st, "active", None)
        if ap is None:
            return
        kind = str(getattr(ap, "kind", "?"))
        rot = int(getattr(ap, "rot", 0))
        px = int(getattr(ap, "x", 0))

        self.max_rots = int(getattr(self.env, "max_rots", self.max_rots))
        self.ghost_rot = int(rot)

        c = self._infer_bbox_left_from_engine_x(kind=kind, rot=int(rot), px=int(px))
        if c is not None:
            self.ghost_col = int(c)

        self._clamp()

    def _infer_bbox_left_from_engine_x(self, *, kind: str, rot: int, px: int) -> Optional[int]:
        for c in range(int(self.board_w)):
            try:
                ex = int(self.legal_cache.bbox_left_to_engine_x(str(kind), int(rot), int(c)))
            except Exception:
                continue
            if int(ex) == int(px):
                return int(c)
        return None

    def _clamp(self) -> None:
        self.max_rots = int(getattr(self.env, "max_rots", self.max_rots))
        self.ghost_rot = int(max(0, min(int(self.ghost_rot), int(self.max_rots - 1))))
        self.ghost_col = int(max(0, min(int(self.ghost_col), int(self.board_w - 1))))

    def _is_legal(self, *, rot: int, col: int) -> bool:
        st = self.game.state() if hasattr(self.game, "state") else self.game._state()  # type: ignore[attr-defined]
        ap = getattr(st, "active", None)
        kind = str(getattr(ap, "kind", "?"))
        py = int(getattr(ap, "y", 0))
        reason = macro_illegal_reason_bbox_left(
            board=self.game.board,
            pieces=self.game.pieces,
            cache=self.legal_cache,
            kind=str(kind),
            rot=int(rot),
            py=int(py),
            bbox_left_col=int(col),
        )
        return bool(reason is None)

    def move_col(self, dx: int) -> None:
        self._clamp()
        c0 = int(self.ghost_col)
        step = 1 if dx >= 0 else -1
        for _ in range(int(self.board_w)):
            c0 += step
            if c0 < 0 or c0 >= int(self.board_w):
                break
            if self._is_legal(rot=int(self.ghost_rot), col=int(c0)):
                self.ghost_col = int(c0)
                return

    def move_rot(self, dr: int) -> None:
        self._clamp()
        r0 = int(self.ghost_rot)
        step = 1 if dr >= 0 else -1
        for _ in range(int(self.max_rots)):
            r0 = (r0 + step) % int(self.max_rots)
            if self._is_legal(rot=int(r0), col=int(self.ghost_col)):
                self.ghost_rot = int(r0)
                return

    def preview_state(self, st: State, *, enabled: bool) -> State:
        if not enabled:
            return st
        ap = getattr(st, "active", None)
        if ap is None:
            return st

        self._clamp()

        kind = str(getattr(ap, "kind", "?"))
        py = int(getattr(ap, "y", 0))

        if not self._is_legal(rot=int(self.ghost_rot), col=int(self.ghost_col)):
            return st

        try:
            px = int(self.legal_cache.bbox_left_to_engine_x(str(kind), int(self.ghost_rot), int(self.ghost_col)))
        except Exception:
            return st

        new_ap = ActivePiece(kind=str(kind), rot=int(self.ghost_rot), x=int(px), y=int(py))
        try:
            return replace(st, active=new_ap)
        except Exception:
            return st

    def action_for_commit(self) -> Any:
        self._clamp()
        action_mode = str(getattr(self.env, "action_mode", "discrete")).strip().lower()
        if action_mode == "discrete":
            return int(self.ghost_rot) * int(self.board_w) + int(self.ghost_col)
        return (int(self.ghost_rot), int(self.ghost_col))
