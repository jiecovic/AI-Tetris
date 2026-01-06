# src/tetris_rl/envs/macro_env.py
from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris_rl.envs.api import RewardFn, WarmupFn
from tetris_rl.envs.illegal_action import (
    IllegalActionPolicy,
    pick_closest_legal_action_id,
    pick_random_legal_action_id,
)
from tetris_rl.envs.macro_actions import ActionMode, MacroActionMixin
from tetris_rl.envs.macro_info import (
    build_reset_info,
    build_step_info_update,
    build_transition_features,
    sidebar_env_rows,
)
from tetris_rl.envs.obs.macro_state import (
    MacroObsSpec,
    build_macro_obs_space,
    encode_macro_obs,
)
from tetris_rl.game.core.macro_legality import discrete_action_mask, macro_illegal_reason_bbox_left
from tetris_rl.game.core.macro_step import decode_discrete_action_id
from tetris_rl.game.core.metrics import BoardSnapshotMetrics, board_snapshot_metrics_from_board
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.game.core.types import State
from tetris_rl.game.core.macro_step import encode_discrete_action_id


class MacroTetrisEnv(MacroActionMixin, gym.Env):
    """
    Macro placement environment.

    North Star:
      - Emits canonical *raw* Dict observations (no tokenization).
      - Tokenization/embedding lives on the policy/model side.
      - Warmup is an injected module (WarmupFn) instead of env init knobs.
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            *,
            game: Any,
            reward_fn: RewardFn,
            max_steps: Optional[int] = None,
            action_mode: ActionMode = "discrete",
            illegal_action_policy: IllegalActionPolicy = "noop",
            warmup: WarmupFn | None = None,
    ) -> None:
        super().__init__()
        self.game = game
        self.reward_fn = reward_fn
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.action_mode = action_mode

        self.illegal_action_policy = str(illegal_action_policy).strip().lower()
        if self.illegal_action_policy not in {"noop", "terminate", "closest_legal", "random_legal"}:
            raise ValueError(f"unknown illegal_action_policy: {illegal_action_policy!r}")

        self._warmup: WarmupFn | None = warmup

        # Board geometry (raw obs). Prefer visible_h if present; else fall back to game.h.
        self.h = int(getattr(game, "visible_h", getattr(game, "h", 20)))
        self.w = int(getattr(game, "w", 10))

        pieces = getattr(self.game, "pieces", None)
        board = getattr(self.game, "board", None)
        active = getattr(self.game, "active", None)
        if pieces is None:
            raise RuntimeError("MacroTetrisEnv requires game.pieces (PieceSet)")
        if board is None:
            raise RuntimeError("MacroTetrisEnv requires game.board (Board)")
        if active is None:
            raise RuntimeError("MacroTetrisEnv requires game.active (ActivePiece)")

        try:
            self.K = int(len(list(pieces.kinds())))
        except Exception as e:
            raise RuntimeError("MacroTetrisEnv cannot determine num_kinds (K) from game.pieces.kinds()") from e
        if self.K <= 0:
            raise RuntimeError("MacroTetrisEnv invalid num_kinds K<=0")

        self._obs_spec = MacroObsSpec(board_h=int(self.h), board_w=int(self.w), num_kinds=int(self.K))
        self.observation_space = build_macro_obs_space(spec=self._obs_spec)

        # Action-space geometry is derived from assets (single source of truth).
        if not hasattr(pieces, "max_rotations"):
            raise RuntimeError("MacroTetrisEnv requires PieceSet.max_rotations() derived from YAML rotations")
        self.max_rots = int(pieces.max_rotations())
        if self.max_rots <= 0:
            raise RuntimeError(f"invalid max_rots derived from assets: {self.max_rots}")

        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(self.max_rots * self.w)
        elif self.action_mode == "multidiscrete":
            self.action_space = spaces.MultiDiscrete([self.max_rots, self.w])
        else:
            raise ValueError(f"unknown action_mode: {self.action_mode!r}")

        self._steps = 0
        self._last_state: State | None = None
        self._episode_idx = 0
        self._prev_board_metrics: BoardSnapshotMetrics | None = None

        self._legal_cache = StaticPlacementCache.build(
            pieces=pieces,
            board_w=int(self.w),
        )

    # ---------------------------------------------------------------------
    # obs
    # ---------------------------------------------------------------------

    def _obs_from_state(self, st: State) -> Dict[str, Any]:
        return encode_macro_obs(
            game=self.game,
            state=st,
            spec=self._obs_spec,
        )

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _piece_rule_name(self) -> str:
        pr = getattr(self.game, "_piece_rule", None)
        if pr is None:
            pr = getattr(self.game, "piece_rule", None)
        if pr is None:
            return "?"
        return getattr(pr.__class__, "__name__", "?")

    def _board_metrics(self) -> BoardSnapshotMetrics | None:
        b = getattr(self.game, "board", None)
        if b is None:
            return None
        try:
            return board_snapshot_metrics_from_board(b)
        except Exception:
            return None

    def _num_rots_for_kind(self, kind: Any) -> int:
        pieces = getattr(self.game, "pieces", None)
        if pieces is None:
            return 1
        try:
            return int(pieces.num_rotations(kind))
        except Exception:
            try:
                return int(pieces.get(kind).num_rotations())
            except Exception:
                return 1

    def _legal_mask_joint_discrete(self, *, kind: str, py: int) -> np.ndarray:
        """
        Compute the joint (max_rots * board_w) legality mask from board+active state.

        IMPORTANT: Valid even when action_mode='multidiscrete' (we just don't expose it
        via sb3-contrib action_masks()).
        """
        return (
            discrete_action_mask(
                board=self.game.board,
                pieces=self.game.pieces,
                kind=str(kind),
                py=int(py),
                cache=self._legal_cache,
            )
            .astype(bool, copy=False)
            .reshape(-1)
        )

    def _illegal_reason_strict(self, *, kind: str, rot: int, col: int, py: int) -> Optional[str]:
        return macro_illegal_reason_bbox_left(
            board=self.game.board,
            pieces=self.game.pieces,
            cache=self._legal_cache,
            kind=str(kind),
            rot=int(rot),
            py=int(py),
            bbox_left_col=int(col),
        )

    def _maybe_warmup_after_reset(self) -> None:
        w = self._warmup
        if w is None:
            return

        # Warmup mutates game state directly (board/grid). Must not record samples.
        w(game=self.game, rng=self.np_random)  # type: ignore[arg-type]

        # Sync cached pointers after warmup mutated the game.
        try:
            self._last_state = self.game._state()  # type: ignore[attr-defined]
        except Exception:
            self._last_state = self.game._state() if hasattr(self.game, "_state") else self._last_state
        self._prev_board_metrics = self._board_metrics()

    # ---------------------------------------------------------------------
    # gym API
    # ---------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._episode_idx += 1

        try:
            self.game.set_rng(self.np_random)  # type: ignore[attr-defined]
        except Exception:
            pass

        st = self.game.reset()
        self._last_state = st
        self._prev_board_metrics = self._board_metrics()

        self._maybe_warmup_after_reset()

        st2 = self._last_state if self._last_state is not None else st
        obs = self._obs_from_state(st2)

        info = build_reset_info(
            state=st2,
            episode_idx=int(self._episode_idx),
            episode_step=int(self._steps),
            action_mode=str(self.action_mode),
            piece_rule=self._piece_rule_name(),
        )
        info["illegal_action_policy"] = str(self.illegal_action_policy)
        info["warmup"] = None if self._warmup is None else getattr(self._warmup.__class__, "__name__", "warmup")
        return obs, info

    def step(self, action: Any):
        self._steps += 1
        prev_state = self._last_state
        if prev_state is None:
            raise RuntimeError("step() called before reset()")

        prev_metrics = self._prev_board_metrics

        act = self._decode_action(action)
        placed_kind = getattr(getattr(prev_state, "active", None), "kind", "?")
        py = int(getattr(getattr(prev_state, "active", None), "y", 0))

        requested_rot = int(act.rot)
        requested_col = int(act.col)

        requested_action_id: Optional[int] = None

        if self.action_mode == "discrete":
            requested_action_id = int(action)
        else:
            requested_action_id = encode_discrete_action_id(
                rot=requested_rot,
                col=requested_col,
                board_w=int(self.w),
            )


        # Mask stats (only meaningful for sb3-contrib MaskablePPO in discrete mode)
        masked_action = False
        action_dim: Optional[int] = None
        masked_action_count: Optional[int] = None
        stats = self.discrete_mask_stats(requested_rot=requested_rot, requested_col=requested_col)
        if stats is not None:
            masked_action = bool(stats.masked_action)
            masked_action_count = stats.masked_action_count
            action_dim = int(stats.action_dim)

        n_rots = max(1, int(self._num_rots_for_kind(placed_kind)))
        redundant_rotation = bool(self.is_redundant_rotation(requested_rot=requested_rot, n_rots_for_kind=n_rots))

        # One truth: strict legality drives illegal_action + illegal_reason.
        info_engine: Any = {}
        illegal_reason = self._illegal_reason_strict(
            kind=str(placed_kind),
            rot=int(requested_rot),
            col=int(requested_col),
            py=int(py),
        )
        illegal_action = bool(illegal_reason is not None)

        mask_mismatch = False
        if self.action_mode == "discrete":
            mask_mismatch = bool(masked_action) != bool(illegal_action)

        remapped = False
        remap_policy: Optional[str] = None

        applied = False
        used_rot = int(requested_rot)
        used_col = int(requested_col)
        cleared_lines = 0
        game_over = False
        st: State = prev_state

        if not illegal_action:
            used_rot, used_col, applied = self._apply_rotation_and_column(rot=requested_rot, col=requested_col)
            if not bool(applied):
                illegal_action = True
                illegal_reason = illegal_reason or "collision"

        if illegal_action:
            remap_policy = str(self.illegal_action_policy)

            if self.illegal_action_policy == "terminate":
                st = prev_state
                cleared_lines = 0
                game_over = True
                applied = False
            elif self.illegal_action_policy == "noop":
                st = prev_state
                cleared_lines = 0
                game_over = False
                applied = False
            else:
                mask = self._legal_mask_joint_discrete(kind=str(placed_kind), py=int(py))

                if self.illegal_action_policy == "random_legal":
                    aid2 = pick_random_legal_action_id(mask, rng=self.np_random)  # type: ignore[arg-type]
                else:
                    aid2 = pick_closest_legal_action_id(
                        mask,
                        requested_rot=int(requested_rot),
                        requested_col=int(requested_col),
                        board_w=int(self.w),
                        max_rots=int(self.max_rots),
                    )

                if aid2 is None:
                    st = prev_state
                    cleared_lines = 0
                    game_over = True
                    applied = False
                else:
                    remapped = True
                    r2, c2 = decode_discrete_action_id(action_id=int(aid2), board_w=int(self.w))
                    used_rot, used_col, applied = self._apply_rotation_and_column(rot=int(r2), col=int(c2))
                    if not bool(applied):
                        st = prev_state
                        cleared_lines = 0
                        game_over = True
                        applied = False
                    else:
                        st, cleared_lines, game_over, info_engine = self.game.step("hard_drop")
        else:
            st, cleared_lines, game_over, info_engine = self.game.step("hard_drop")

        # engine-derived: how many placed cells vanished
        placed_cells_cleared = 0
        placed_all_cells_cleared = False
        if isinstance(info_engine, dict):
            try:
                placed_cells_cleared = int(info_engine.get("placed_cells_cleared", 0))
            except Exception:
                placed_cells_cleared = 0
            placed_all_cells_cleared = bool(info_engine.get("placed_all_cells_cleared", False))

        # metrics + deltas
        cur_metrics = self._board_metrics()

        holes_after: Optional[int] = None
        max_height_after: Optional[int] = None
        bumpiness_after: Optional[int] = None
        agg_height_after: Optional[int] = None

        delta_holes: Optional[int] = None
        delta_max_height: Optional[int] = None
        delta_bumpiness: Optional[int] = None
        delta_agg_height: Optional[int] = None

        if cur_metrics is not None:
            holes_after = int(cur_metrics.holes)
            max_height_after = int(cur_metrics.max_height)
            bumpiness_after = int(cur_metrics.bumpiness)
            agg_height_after = int(cur_metrics.agg_height)

        if prev_metrics is not None and cur_metrics is not None:
            delta_holes = int(cur_metrics.holes - prev_metrics.holes)
            delta_max_height = int(cur_metrics.max_height - prev_metrics.max_height)
            delta_bumpiness = int(cur_metrics.bumpiness - prev_metrics.bumpiness)
            delta_agg_height = int(cur_metrics.agg_height - prev_metrics.agg_height)

        self._prev_board_metrics = cur_metrics

        cleared = int(cleared_lines)
        terminated = bool(game_over)
        truncated = bool(self.max_steps is not None and self._steps >= self.max_steps and not terminated)

        # env-only: delta_score
        prev_score = float(getattr(prev_state, "score", 0.0))
        next_score = float(getattr(st, "score", 0.0))
        delta_score = float(next_score - prev_score)

        features = build_transition_features(
            cleared=cleared,
            placed_cells_cleared=int(placed_cells_cleared),
            placed_all_cells_cleared=bool(placed_all_cells_cleared),
            terminated=terminated,
            placed_kind=str(placed_kind),
            requested_rot=int(requested_rot),
            requested_col=int(requested_col),
            used_rot=int(used_rot),
            used_col=int(used_col),
            applied=bool(applied),
            illegal_action=bool(illegal_action),
            illegal_reason=illegal_reason,
            remapped=bool(remapped),
            remap_policy=remap_policy,
            masked_action=bool(masked_action),
            redundant_rotation=bool(redundant_rotation),
            delta_holes=delta_holes,
            delta_max_height=delta_max_height,
            delta_bumpiness=delta_bumpiness,
            delta_agg_height=delta_agg_height,
            holes_after=holes_after,
            max_height_after=max_height_after,
            bumpiness_after=bumpiness_after,
            agg_height_after=agg_height_after,
        )

        sidebar_env = sidebar_env_rows(
            illegal_action=bool(illegal_action),
            redundant_rotation=bool(redundant_rotation),
            placed_cells_cleared=int(placed_cells_cleared),
            placed_all_cells_cleared=bool(placed_all_cells_cleared),
            delta_holes=delta_holes,
            delta_max_height=delta_max_height,
            delta_bumpiness=delta_bumpiness,
            bumpiness=bumpiness_after,
        )

        info = build_step_info_update(
            # env truth
            illegal_action=bool(illegal_action),
            illegal_reason=illegal_reason,
            illegal_action_policy=str(self.illegal_action_policy),
            remapped=bool(remapped),
            remap_policy=remap_policy,
            applied=bool(applied),
            mask_mismatch=bool(mask_mismatch),
            game_over=bool(terminated),
            delta_score=float(delta_score),
            # state/presentation
            state=st,
            cleared=cleared,
            placed_cells_cleared=int(placed_cells_cleared),
            placed_all_cells_cleared=bool(placed_all_cells_cleared),
            action_mode=str(self.action_mode),
            requested_rot=int(requested_rot),
            requested_col=int(requested_col),
            requested_action_id=int(requested_action_id),
            used_rot=int(used_rot),
            used_col=int(used_col),
            masked_action=bool(masked_action),
            redundant_rotation=bool(redundant_rotation),
            action_dim=action_dim,
            masked_action_count=masked_action_count,
            episode_idx=int(self._episode_idx),
            episode_step=int(self._steps),
            piece_rule=self._piece_rule_name(),
            holes_after=holes_after,
            delta_holes=delta_holes,
            max_height_after=max_height_after,
            delta_max_height=delta_max_height,
            bumpiness_after=bumpiness_after,
            delta_bumpiness=delta_bumpiness,
            agg_height_after=agg_height_after,
            delta_agg_height=delta_agg_height,
            sidebar_env=sidebar_env,
            # engine passthrough
            engine_info=info_engine if isinstance(info_engine, dict) else None,
        )

        shaped = float(
            self.reward_fn(
                prev_state=prev_state,
                action=action,
                next_state=st,
                features=features,
                info=info,
            )
        )

        self._last_state = st
        obs = self._obs_from_state(st)
        return obs, shaped, terminated, truncated, info

    @property
    def last_state(self) -> State:
        st = self._last_state
        if st is None:
            raise RuntimeError("MacroTetrisEnv.last_state accessed before reset()")
        return st
