# src/tetris_rl/envs/macro_actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np

from tetris_rl.game.core.macro_legality import discrete_action_mask
from tetris_rl.game.core.macro_step import (
    decode_discrete_action_id,
    encode_discrete_action_id,
    try_apply_rotation_and_bbox_left_column_strict,
)
from tetris_rl.game.core.placement_cache import StaticPlacementCache

ActionMode = Literal["discrete", "multidiscrete"]


@dataclass(frozen=True)
class MacroAction:
    rot: int
    col: int


@dataclass(frozen=True)
class DiscreteMaskStats:
    masked_action: bool
    masked_action_count: Optional[int]
    action_dim: int


class MacroActionMixin:
    """
    Action decoding, STRICT placement application, and (optional) action masking.

    Expects the concrete env to define:
      - self.action_mode: ActionMode
      - self.w: int
      - self.max_rots: int
      - self.game: has pieces/board/active
      - self._legal_cache: StaticPlacementCache
    """

    action_mode: ActionMode
    w: int
    max_rots: int
    game: Any
    _legal_cache: StaticPlacementCache

    def action_masks(self) -> np.ndarray:
        """
        sb3-contrib MaskablePPO hook.

        Returns boolean mask of shape (action_space.n,), where True means "legal".
        Only supported for action_mode='discrete' (joint rot×col actions).
        """
        if self.action_mode != "discrete":
            raise RuntimeError(
                "action_masks() is only supported for action_mode='discrete'. "
                "MultiDiscrete masking is not supported (joint constraints)."
            )

        ap = self.game.active
        return discrete_action_mask(
            board=self.game.board,
            pieces=self.game.pieces,
            cache=self._legal_cache,
            kind=str(ap.kind),
            py=int(ap.y),
        )

    def discrete_mask_stats(self, *, requested_rot: int, requested_col: int) -> Optional[DiscreteMaskStats]:
        """
        Diagnostics for the *requested* (rot,col) under joint Discrete masking.

        Returns None if not in discrete mode or mask is unavailable/malformed.
        """
        if self.action_mode != "discrete":
            return None

        bw = int(self.w)
        mr = int(getattr(self, "max_rots", 0) or 0)
        if bw <= 0 or mr <= 0:
            return None

        action_dim = int(mr * bw)

        try:
            a_req = encode_discrete_action_id(rot=int(requested_rot), col=int(requested_col), board_w=bw)
        except Exception:
            return None

        try:
            m = self.action_masks()
            mask = np.asarray(m, dtype=bool).reshape(-1)
        except Exception:
            return None

        if int(mask.shape[0]) != action_dim:
            return None

        masked_action_count = int((~mask).sum())
        masked_action = bool(0 <= int(a_req) < int(action_dim) and not bool(mask[int(a_req)]))

        return DiscreteMaskStats(
            masked_action=bool(masked_action),
            masked_action_count=int(masked_action_count),
            action_dim=int(action_dim),
        )

    @staticmethod
    def is_redundant_rotation(*, requested_rot: int, n_rots_for_kind: int) -> bool:
        """
        True iff requested_rot is outside the asset-valid rotations for this kind.

        Under STRICT semantics, redundant rotations are simply illegal actions.
        """
        n = max(1, int(n_rots_for_kind))
        return bool(int(requested_rot) >= n)

    def _decode_action(self, a: Any) -> MacroAction:
        if self.action_mode == "discrete":
            rot, col = decode_discrete_action_id(action_id=int(a), board_w=int(self.w))
            return MacroAction(rot=int(rot), col=int(col))

        if isinstance(a, (tuple, list)) and len(a) == 2:
            rot, col = a
            return MacroAction(rot=int(rot), col=int(col))

        arr = np.asarray(a).reshape(-1)
        if arr.size == 2:
            return MacroAction(rot=int(arr[0]), col=int(arr[1]))

        raise TypeError(f"invalid action for action_mode='multidiscrete': {a!r}")

    def _apply_rotation_and_column(self, *, rot: int, col: int) -> Tuple[int, int, bool]:
        """
        STRICT application (North Star).
        Returns (used_rot, used_col, applied).
        """
        return try_apply_rotation_and_bbox_left_column_strict(
            game=self.game,
            legal_cache=self._legal_cache,
            rot=int(rot),
            col=int(col),
        )
