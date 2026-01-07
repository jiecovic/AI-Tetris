# src/tetris_rl/envs/macro_env.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

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


def _encode_action_id(*, rot: int, col: int, board_w: int, max_rots: int) -> int:
    # Fixed layout: rot-major, then columns.
    r = int(rot) % int(max_rots)
    c = int(col)
    return r * int(board_w) + c


def _decode_action_id(*, action_id: int, board_w: int) -> tuple[int, int]:
    aid = int(action_id)
    w = int(board_w)
    r = aid // w
    c = aid - r * w
    return int(r), int(c)


def _kind_num_rots_classic7(kind_glyph: str) -> int:
    # Only for redundant-rotation UI/debug. Engine legality is authoritative anyway.
    k = str(kind_glyph).upper()
    if k == "O":
        return 1
    if k in {"I", "S", "Z"}:
        return 2
    return 4  # T,J,L


class MacroTetrisEnv(MacroActionMixin, gym.Env):
    """
    Macro placement environment backed by the Rust engine (PyO3).

    - Observations are canonical raw Dicts (no tokenization).
    - Legality is driven by engine.action_mask() (single source of truth).
    - State is a dict snapshot from engine.snapshot(...).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        game: Any,  # PyO3 TetrisEngine
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

        # Warmup: preferred path is passing a Rust WarmupSpec into engine.reset(...).
        # We keep WarmupFn as an injection point; it should return a PyWarmupSpec (or None).
        self._warmup: WarmupFn | None = warmup

        # Geometry from engine (single source of truth).
        self.h = int(self.game.visible_h())
        self.w = int(self.game.board_w())
        self.max_rots = int(self.game.max_rots())
        self.action_dim = int(self.game.action_dim())

        # K: classic 7. (If you later generalize pieces, expose num_kinds from Rust.)
        self.K = 7

        self._obs_spec = MacroObsSpec(board_h=int(self.h), board_w=int(self.w), num_kinds=int(self.K))
        self.observation_space = build_macro_obs_space(spec=self._obs_spec)

        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(int(self.action_dim))
        elif self.action_mode == "multidiscrete":
            self.action_space = spaces.MultiDiscrete([int(self.max_rots), int(self.w)])
        else:
            raise ValueError(f"unknown action_mode: {self.action_mode!r}")

        self._steps = 0
        self._episode_idx = 0

        # last snapshot dict from engine.snapshot(...)
        self._last_state: Dict[str, Any] | None = None

        # previous grid features tuple (max_h, agg_h, holes, bump) for deltas
        self._prev_feat: Optional[Tuple[int, int, int, int]] = None

    # ---------------------------------------------------------------------
    # obs
    # ---------------------------------------------------------------------

    def _obs_from_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        return encode_macro_obs(game=None, state=st, spec=self._obs_spec)

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _piece_rule_name(self) -> str:
        # Engine exposes PieceRuleKind only in Rust; keep best-effort.
        return "rust_engine"

    def _action_mask_bool(self) -> np.ndarray:
        m_u8 = np.asarray(self.game.action_mask(), dtype=np.uint8)
        # (ACTION_DIM,) where 1=legal
        return (m_u8.astype(np.uint8, copy=False) != 0).reshape(-1)

    def _mask_stats_for_requested(self, requested_action_id: int) -> tuple[bool, int, int]:
        mask = self._action_mask_bool()
        aid = int(requested_action_id)
        masked = bool(aid < 0 or aid >= mask.size or (not bool(mask[aid])))
        masked_count = int((~mask).sum())
        return masked, masked_count, int(mask.size)

    def _episode_seed_from_np_random(self) -> int:
        # Deterministic per-env RNG -> deterministic episodes if outer seed fixed.
        return int(self.np_random.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))

    def _warmup_spec_for_reset(self) -> Any:
        if self._warmup is None:
            return None
        # WarmupFn is user-injected; convention here: returns PyWarmupSpec or None.
        return self._warmup(game=self.game, rng=self.np_random)  # type: ignore[misc]

    # ---------------------------------------------------------------------
    # sb3-contrib MaskablePPO hook
    # ---------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        # sb3-contrib expects bool mask for Discrete.
        return self._action_mask_bool()

    # ---------------------------------------------------------------------
    # gym API
    # ---------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._episode_idx += 1
        self._prev_feat = None

        ep_seed = int(seed) if seed is not None else self._episode_seed_from_np_random()
        warmup_spec = self._warmup_spec_for_reset()

        # Engine requires explicit seed.
        self.game.reset(seed=int(ep_seed), warmup=warmup_spec)

        st = cast(Dict[str, Any], self.game.snapshot(include_grid=True, visible=True))
        self._last_state = st

        obs = self._obs_from_state(st)

        info = build_reset_info(
            state=st,
            episode_idx=int(self._episode_idx),
            episode_step=int(self._steps),
            action_mode=str(self.action_mode),
            piece_rule=self._piece_rule_name(),
        )
        info["illegal_action_policy"] = str(self.illegal_action_policy)
        info["warmup"] = None if self._warmup is None else getattr(self._warmup.__class__, "__name__", "warmup")
        info["episode_seed"] = int(ep_seed)
        return obs, info

    def step(self, action: Any):
        self._steps += 1
        prev_state = self._last_state
        if prev_state is None:
            raise RuntimeError("step() called before reset()")

        # Decode action to (rot,col) + action_id
        act = self._decode_action(action)
        requested_rot = int(act.rot)
        requested_col = int(act.col)

        if self.action_mode == "discrete":
            requested_action_id = int(action)
        else:
            requested_action_id = _encode_action_id(
                rot=requested_rot,
                col=requested_col,
                board_w=int(self.w),
                max_rots=int(self.max_rots),
            )

        # Mask stats (for UI/debug + MaskablePPO mismatch check)
        masked_action, masked_action_count, action_dim = self._mask_stats_for_requested(requested_action_id)

        # Redundant rotation (UI/debug only)
        placed_kind = str(prev_state.get("active_kind", "?"))
        n_rots = _kind_num_rots_classic7(placed_kind)
        redundant_rotation = bool(self.is_redundant_rotation(requested_rot=requested_rot, n_rots_for_kind=int(n_rots)))

        # Engine legality is authoritative.
        mask = self._action_mask_bool()
        illegal_action = bool(
            requested_action_id < 0
            or requested_action_id >= int(mask.size)
            or (not bool(mask[int(requested_action_id)]))
        )
        illegal_reason: Optional[str] = "masked" if illegal_action else None

        mask_mismatch = False
        if self.action_mode == "discrete":
            mask_mismatch = bool(masked_action) != bool(illegal_action)

        remapped = False
        remap_policy: Optional[str] = None
        used_action_id = int(requested_action_id)

        terminated = False
        truncated = False
        cleared = 0
        info_engine: Dict[str, Any] = {}

        if illegal_action:
            remap_policy = str(self.illegal_action_policy)

            if self.illegal_action_policy == "terminate":
                terminated = True
                cleared = 0
                info_engine = {}
            elif self.illegal_action_policy == "noop":
                terminated = False
                cleared = 0
                info_engine = {}
            else:
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
                    terminated = True
                    cleared = 0
                    info_engine = {}
                else:
                    remapped = True
                    used_action_id = int(aid2)
                    terminated, cleared, _illegal2 = self.game.step_action_id(int(used_action_id))
                    info_engine = {}
        else:
            terminated, cleared, _illegal = self.game.step_action_id(int(used_action_id))
            info_engine = {}

        if self.max_steps is not None and self._steps >= self.max_steps and not bool(terminated):
            truncated = True

        # New snapshot after step (include grid for obs)
        st = cast(Dict[str, Any], self.game.snapshot(include_grid=True, visible=True))
        self._last_state = st

        # Features (holes/max_h/bump/agg + deltas) from Rust
        sf = cast(Dict[str, Any], self.game.step_features(prev=self._prev_feat, visible=False))
        cur = cast(Dict[str, Any], sf.get("cur", {}))
        delta = cast(Dict[str, Any], sf.get("delta", {}))

        holes_after = int(cur.get("holes", 0)) if cur else None
        max_height_after = int(cur.get("max_h", 0)) if cur else None
        bumpiness_after = int(cur.get("bump", 0)) if cur else None
        agg_height_after = int(cur.get("agg_h", 0)) if cur else None

        delta_holes = int(delta.get("d_holes", 0)) if delta else None
        delta_max_height = int(delta.get("d_max_h", 0)) if delta else None
        delta_bumpiness = int(delta.get("d_bump", 0)) if delta else None
        delta_agg_height = int(delta.get("d_agg_h", 0)) if delta else None

        if cur:
            self._prev_feat = (
                int(cur.get("max_h", 0)),
                int(cur.get("agg_h", 0)),
                int(cur.get("holes", 0)),
                int(cur.get("bump", 0)),
            )

        # No longer available from engine: keep 0/False for now (contract remains).
        placed_cells_cleared = 0
        placed_all_cells_cleared = False

        # env-only: delta_score from snapshots
        prev_score = float(prev_state.get("score", 0.0))
        next_score = float(st.get("score", 0.0))
        delta_score = float(next_score - prev_score)

        used_rot, used_col = _decode_action_id(action_id=int(used_action_id), board_w=int(self.w))

        features = build_transition_features(
            cleared=int(cleared),
            placed_cells_cleared=int(placed_cells_cleared),
            placed_all_cells_cleared=bool(placed_all_cells_cleared),
            terminated=bool(terminated),
            placed_kind=str(placed_kind),
            requested_rot=int(requested_rot),
            requested_col=int(requested_col),
            used_rot=int(used_rot),
            used_col=int(used_col),
            applied=bool(not illegal_action) or bool(remapped),
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
            illegal_action=bool(illegal_action),
            illegal_reason=illegal_reason,
            illegal_action_policy=str(self.illegal_action_policy),
            remapped=bool(remapped),
            remap_policy=remap_policy,
            applied=bool(not illegal_action) or bool(remapped),
            mask_mismatch=bool(mask_mismatch),
            game_over=bool(terminated),
            delta_score=float(delta_score),
            state=st,
            cleared=int(cleared),
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
            action_dim=int(action_dim),
            masked_action_count=int(masked_action_count),
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
            engine_info=info_engine,
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

        obs = self._obs_from_state(st)
        return obs, shaped, bool(terminated), bool(truncated), info

    @property
    def last_state(self) -> Dict[str, Any]:
        st = self._last_state
        if st is None:
            raise RuntimeError("MacroTetrisEnv.last_state accessed before reset()")
        return st
