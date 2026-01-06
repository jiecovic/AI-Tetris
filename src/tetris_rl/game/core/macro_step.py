# src/tetris_rl/game/core/macro_step.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from tetris_rl.game.core.macro_legality import macro_illegal_reason_bbox_left
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.game.core.types import ActivePiece, State


@dataclass(frozen=True)
class MacroApplyResult:
    state: State
    cleared_lines: int
    terminated: bool
    used_rot: int
    used_col: int
    applied: bool  # whether the requested (rot,col) was actually applied (strict legality)

    # Engine diagnostics / extensions (e.g. cleared rows, placed cells)
    info_engine: Dict[str, Any]


def encode_discrete_action_id(*, rot: int, col: int, board_w: int) -> int:
    bw = int(board_w)
    if bw <= 0:
        raise ValueError(f"board_w must be positive, got {bw}")
    return int(int(rot) * bw + int(col))


def decode_discrete_action_id(*, action_id: int, board_w: int) -> Tuple[int, int]:
    bw = int(board_w)
    if bw <= 0:
        raise ValueError(f"board_w must be positive, got {bw}")
    ai = int(action_id)
    return int(ai // bw), int(ai % bw)


def try_apply_rotation_and_bbox_left_column_strict(
    *,
    game: Any,
    legal_cache: StaticPlacementCache,
    rot: int,
    col: int,
) -> Tuple[int, int, bool]:
    """
    STRICT North-Star application.

    Legal iff (single ground truth):
      - rotation exists for the piece kind
      - bbox-left col is in bounds for that rotation
      - placement at current spawn/active py does NOT collide with the locked board

    No wrap. No clamp. No silent fixes.
    Returns (used_rot, used_col, applied).
    """
    ap = game.active
    kind = str(ap.kind)

    r = int(rot)
    c = int(col)
    py = int(ap.y)

    reason = macro_illegal_reason_bbox_left(
        board=game.board,
        pieces=game.pieces,
        cache=legal_cache,
        kind=kind,
        rot=r,
        py=py,
        bbox_left_col=c,
    )
    if reason is not None:
        # active piece unchanged
        return int(ap.rot), int(c), False

    px = int(legal_cache.bbox_left_to_engine_x(kind, r, c))
    new_ap = ActivePiece(kind=kind, rot=int(r), x=int(px), y=int(py))
    game.active = new_ap
    return int(new_ap.rot), int(c), True


def apply_discrete_action_id_no_reward_with_diag(
    *,
    game: Any,
    legal_cache: StaticPlacementCache,
    action_id: int,
    board_w: int,
) -> MacroApplyResult:
    rot, col = decode_discrete_action_id(action_id=int(action_id), board_w=int(board_w))

    used_rot, used_col, applied = try_apply_rotation_and_bbox_left_column_strict(
        game=game,
        legal_cache=legal_cache,
        rot=int(rot),
        col=int(col),
    )

    st, cleared_lines, game_over, info_engine = game.step("hard_drop")

    info_dict: Dict[str, Any] = {}
    if isinstance(info_engine, dict):
        info_dict.update(info_engine)

    return MacroApplyResult(
        state=st,
        cleared_lines=int(cleared_lines),
        terminated=bool(game_over),
        used_rot=int(used_rot),
        used_col=int(used_col),
        applied=bool(applied),
        info_engine=info_dict,
    )


def apply_discrete_action_id_no_reward(
    *,
    game: Any,
    legal_cache: StaticPlacementCache,
    action_id: int,
    board_w: int,
) -> Tuple[State, int, bool]:
    """
    Backwards-compatible wrapper: uses the STRICT path but drops diagnostics.
    """
    r = apply_discrete_action_id_no_reward_with_diag(
        game=game,
        legal_cache=legal_cache,
        action_id=int(action_id),
        board_w=int(board_w),
    )
    return r.state, int(r.cleared_lines), bool(r.terminated)
