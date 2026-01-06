# src/tetris_rl/game/core/game.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from tetris_rl.game.core.board import Board
from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.game.core.piece_rules import PieceRule, UniformPieceRule
from tetris_rl.game.core.rotation import collides, try_rotate
from tetris_rl.game.core.rules import ScoreConfig, score_for_clears
from tetris_rl.game.core.types import Action, ActivePiece, State


class TetrisGame:
    """
    Minimal playable engine.

    Contracts:

      - board.grid is the authoritative LOCKED board.
      - reset()/step() return a State whose grid is the LOCKED board grid (NO COPY).
      - Rendering must overlay the active piece in the renderer layer.
      - step() returns cleared_lines (0..4) as a game event metric, NOT an RL reward.
      - A Gymnasium env should inject its RNG via set_rng(env.np_random) before reset().

      - SPECIAL token contract (strict, no legacy fallback):
          State.active_kind_idx / State.next_kind_idx are required and are kind indices {0..K-1}.

      - BOARD cell encoding contract (categorical board):
          board_id in grid is:
            0 = empty
            1..K = kind_idx + 1
        This is independent of the special-token encoding and is what board tokenizers read.
    """

    def __init__(
            self,
            *,
            visible_height: int = 20,
            spawn_rows: int = 2,
            width: int = 10,
            piece_set: Optional[PieceSet] = None,
            piece_rule: PieceRule | None = None,
    ) -> None:
        self.visible_h = int(visible_height)
        self.spawn_rows = int(spawn_rows)
        if self.visible_h <= 0:
            raise ValueError(f"visible_height must be positive, got {self.visible_h}")
        if self.spawn_rows < 0:
            raise ValueError(f"spawn_rows must be >= 0, got {self.spawn_rows}")

        self.h = int(self.visible_h + self.spawn_rows)
        self.w = int(width)
        if self.w <= 0:
            raise ValueError(f"width must be positive, got {self.w}")

        self.pieces = piece_set or PieceSet.from_yaml(PieceSet.default_classic7_path(), expected_cells=4)

        kinds = list(self.pieces.kinds())
        if not kinds:
            raise ValueError("PieceSet has no kinds (empty pieceset is invalid).")

        # Canonical kind-index mapping (0..K-1), derived from PieceSet.kinds() order.
        self._kinds: tuple[str, ...] = tuple(str(k) for k in kinds)
        self._kind_to_idx: dict[str, int] = {k: i for i, k in enumerate(self._kinds)}

        self.board = Board.empty(h=self.h, w=self.w)
        self.score_cfg = ScoreConfig()

        # Env-owned RNG (Gymnasium). We keep a placeholder generator for standalone usage.
        self._rng: np.random.Generator = np.random.default_rng()

        # Piece selection rule (uniform by default).
        self._piece_rule: PieceRule = piece_rule or UniformPieceRule()

        # next-piece preview: initialized on reset()
        self.next_kind: str = str(self._kinds[0])
        self.active = ActivePiece(kind=self.next_kind, rot=0, x=(self.w // 2) - 2, y=0)

        # Used by GB-style rules that depend on last locked kind.
        self._last_locked_kind: str | None = None

        self.score = 0
        self.lines = 0
        self.level = 0
        self.game_over = False

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def reset(self) -> State:
        self.board = Board.empty(h=self.h, w=self.w)
        self.score = 0
        self.lines = 0
        self.level = 0
        self.game_over = False
        self._last_locked_kind = None

        kinds = list(self.pieces.kinds())
        if not kinds:
            raise ValueError("PieceSet has no kinds (empty pieceset is invalid).")

        # Re-freeze mapping in case PieceSet changed (should not in practice).
        self._kinds = tuple(str(k) for k in kinds)
        self._kind_to_idx = {k: i for i, k in enumerate(self._kinds)}

        self._piece_rule.reset(rng=self._rng, kinds=kinds)

        # Initialize preview, then spawn active from preview.
        nk = self._piece_rule.next_piece(locked_kind=None, preview_kind=None)
        if nk is None:
            raise RuntimeError("PieceRule.next_piece returned None (invalid).")
        self.next_kind = str(nk)

        self._spawn()
        return self._state()

    def step(self, action: Any) -> Tuple[State, int, bool, Dict[str, object]]:
        """
        Apply an action and return:

          (state, cleared_lines, game_over, info)

        cleared_lines is a game event metric (0..4). It is NOT an RL reward.
        """
        if self.game_over:
            return self._state(), 0, True, {}

        a = self._normalize_action(action)
        cleared_lines = 0
        info: Dict[str, object] = {}

        if a == Action.LEFT:
            self._try_move(dx=-1, dy=0)
        elif a == Action.RIGHT:
            self._try_move(dx=+1, dy=0)
        elif a == Action.SOFT_DROP:
            if not self._try_move(dx=0, dy=+1):
                c, placed_cleared, placed_all = self._lock_and_advance()
                cleared_lines += int(c)
                info["placed_cells_cleared"] = int(placed_cleared)
                info["placed_all_cells_cleared"] = bool(placed_all)
        elif a == Action.HARD_DROP:
            while self._try_move(dx=0, dy=+1):
                pass
            c, placed_cleared, placed_all = self._lock_and_advance()
            cleared_lines += int(c)
            info["placed_cells_cleared"] = int(placed_cleared)
            info["placed_all_cells_cleared"] = bool(placed_all)
        elif a == Action.ROT_CW:
            self._rotate(dir=+1)
        elif a == Action.ROT_CCW:
            self._rotate(dir=-1)
        elif a == Action.HOLD:
            info["hold"] = "not_implemented"

        return self._state(), int(cleared_lines), bool(self.game_over), info

    # ---- internals -----------------------------------------------------------------

    def _normalize_action(self, action: Any) -> Action:
        if isinstance(action, Action):
            return action
        s = str(action).lower()
        mapping = {
            "left": Action.LEFT,
            "right": Action.RIGHT,
            "soft_drop": Action.SOFT_DROP,
            "down": Action.SOFT_DROP,
            "hard_drop": Action.HARD_DROP,
            "drop": Action.HARD_DROP,
            "rot_cw": Action.ROT_CW,
            "rotate_cw": Action.ROT_CW,
            "rotate_right": Action.ROT_CW,
            "cw": Action.ROT_CW,
            "rot_ccw": Action.ROT_CCW,
            "rotate_ccw": Action.ROT_CCW,
            "rotate_left": Action.ROT_CCW,
            "ccw": Action.ROT_CCW,
            "hold": Action.HOLD,
        }
        return mapping.get(s, Action.SOFT_DROP)

    def _spawn(self) -> None:
        # Promote preview -> active, then sample new preview using the rule.
        kind = self.next_kind

        nk = self._piece_rule.next_piece(
            locked_kind=self._last_locked_kind,
            preview_kind=kind,
        )
        if nk is None:
            raise RuntimeError("PieceRule.next_piece returned None (invalid).")
        self.next_kind = str(nk)

        ap = ActivePiece(kind=kind, rot=0, x=(self.w // 2) - 2, y=0)
        if collides(board=self.board, pieces=self.pieces, kind=ap.kind, rot=ap.rot, px=ap.x, py=ap.y):
            self.game_over = True
        self.active = ap

    def _try_move(self, dx: int, dy: int) -> bool:
        ap = self.active
        nx, ny = ap.x + dx, ap.y + dy
        if collides(board=self.board, pieces=self.pieces, kind=ap.kind, rot=ap.rot, px=nx, py=ny):
            return False
        self.active = ActivePiece(kind=ap.kind, rot=ap.rot, x=nx, y=ny)
        return True

    def _rotate(self, dir: int) -> None:
        ap = self.active
        nrot = try_rotate(
            board=self.board,
            pieces=self.pieces,
            kind=ap.kind,
            rot=ap.rot,
            px=ap.x,
            py=ap.y,
            dir=dir,
        )
        self.active = ActivePiece(kind=ap.kind, rot=nrot, x=ap.x, y=ap.y)

    def _is_lock_out(self) -> bool:
        sr = int(self.spawn_rows)
        if sr <= 0:
            return False
        try:
            return bool(np.any(self.board.grid[:sr, :] != 0))
        except Exception:
            for y in range(min(sr, self.h)):
                for x in range(self.w):
                    if int(self.board.grid[y, x]) != 0:
                        return True
            return False

    def _kind_idx(self, kind: str) -> int:
        """
        Strict kind->index mapping: kind_idx ∈ {0..K-1}.
        """
        try:
            return int(self._kind_to_idx[kind])
        except KeyError as e:
            raise KeyError(f"unknown kind {kind!r} (not in PieceSet kinds={list(self._kinds)!r})") from e

    def _board_id(self, kind: str) -> int:
        """
        Categorical board id: board_id ∈ {1..K} with 0 reserved for empty.
        """
        return int(self._kind_idx(kind) + 1)

    def _lock_and_advance(self) -> Tuple[int, int, bool]:
        """
        Lock the active piece into the board, clear lines, update score/lines, then spawn next.

        Returns:
          (cleared_lines, placed_cells_cleared, placed_all_cells_cleared)

        placed_cells_cleared:
          number of tetromino cells that vanished due to line clears (0..4)

        placed_all_cells_cleared:
          True iff the whole tetromino vanished due to line clears
        """
        ap = self.active
        self._last_locked_kind = ap.kind

        m = self.pieces.mask(ap.kind, ap.rot)

        # Board stores categorical ids: 0=empty, 1..K=kind_idx+1.
        board_id = self._board_id(ap.kind)

        placed_cells: list[tuple[int, int]] = []

        h, w = m.shape
        for yy in range(h):
            for xx in range(w):
                if m[yy, xx] == 0:
                    continue
                x = ap.x + xx
                y = ap.y + yy
                if 0 <= x < self.w and 0 <= y < self.h:
                    placed_cells.append((int(x), int(y)))
                    self.board.grid[y, x] = board_id

        # Identify which rows will be cleared (based on the locked board *before* clearing).
        # This avoids modifying Board.clear_full_lines() API.
        try:
            full = np.all(self.board.grid != 0, axis=1)
            cleared_rows = set(np.flatnonzero(full).astype(int).tolist())
        except Exception:
            cleared_rows = set()
            for y in range(self.h):
                ok = True
                for x in range(self.w):
                    if int(self.board.grid[y, x]) == 0:
                        ok = False
                        break
                if ok:
                    cleared_rows.add(int(y))

        placed_cells_cleared = 0
        if placed_cells and cleared_rows:
            placed_cells_cleared = int(sum(1 for (_x, y) in placed_cells if int(y) in cleared_rows))
        placed_all_cells_cleared = bool(
            placed_cells_cleared == int(len(placed_cells)) and int(len(placed_cells)) > 0
        )

        cleared = int(self.board.clear_full_lines())
        self.lines += cleared

        # Level rule (simple): 10 lines -> +1 level
        # Monotone and deterministic.
        new_level = int(self.lines) // 10
        if new_level > int(self.level):
            self.level = int(new_level)

        self.score += score_for_clears(cleared, self.score_cfg)

        if self._is_lock_out():
            self.game_over = True
            return int(cleared), int(placed_cells_cleared), bool(placed_all_cells_cleared)

        self._spawn()
        return int(cleared), int(placed_cells_cleared), bool(placed_all_cells_cleared)

    def _state(self) -> State:
        """
        Engine snapshot for env/renderer:

        - grid is the LOCKED board grid (no copy, no active overlay)
        - active piece is provided separately in State.active
        - specials are strict kind indices (0..K-1)
        """
        ap = self.active
        nk = str(self.next_kind)

        return State(
            grid=self.board.grid,
            score=int(self.score),
            lines=int(self.lines),
            level=int(self.level),
            game_over=bool(self.game_over),
            active=ap,
            next_kind=nk,
            active_kind_idx=int(self._kind_idx(ap.kind)),
            next_kind_idx=int(self._kind_idx(nk)),
            active_rot=int(ap.rot),
        )

    def state(self) -> State:
        """
        Public, stable snapshot accessor.

        Returns the same snapshot as the engine internals:
          - grid is the LOCKED board grid (no copy, no active overlay)
          - active is provided separately
          - specials are strict kind indices (0..K-1)

        This method is the supported API for env/datagen/renderer.
        """
        return self._state()
