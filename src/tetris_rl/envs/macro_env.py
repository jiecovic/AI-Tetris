# src/tetris_rl/env_bundles/macro_env.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris_rl.envs.api import RewardFn, WarmupFn
from tetris_rl.envs.invalid_action import InvalidActionPolicy
from tetris_rl.envs.macro_actions import ActionMode
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


class MacroTetrisEnv(gym.Env):
    """
    Macro placement environment backed by the Rust engine (PyO3).

    SSOT:
      - Validity: engine.step_action_id() return value.
      - Mask: MaskablePPO action_masks() + optional mismatch debug.
      - Action encoding/decoding: engine.encode_action_id / engine.decode_action_id.
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            *,
            game: Any,  # PyO3 TetrisEngine
            reward_fn: RewardFn,
            max_steps: Optional[int] = None,
            action_mode: ActionMode = "discrete",
            invalid_action_policy: InvalidActionPolicy = "noop",
            warmup: WarmupFn | None = None,
    ) -> None:
        super().__init__()
        self.game = game
        self.reward_fn = reward_fn
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.action_mode = action_mode

        self.invalid_action_policy = str(invalid_action_policy).strip().lower()
        if self.invalid_action_policy not in {"noop", "terminate"}:
            raise ValueError(f"unknown invalid_action_policy: {invalid_action_policy!r}")

        self._warmup: WarmupFn | None = warmup

        # Geometry from engine (SSOT).
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

        self._last_state: Dict[str, Any] | None = None
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
        # Preferred: call into engine if available.
        try:
            return str(self.game.piece_rule())
        except Exception:
            # Fallback: snapshot key (if present), else "unknown"
            st = self._last_state
            return str(st.get("piece_rule", "unknown")) if st else "unknown"

    def _action_mask_bool(self) -> np.ndarray:
        m_u8 = np.asarray(self.game.action_mask(), dtype=np.uint8)
        # (ACTION_DIM,) where 1=valid, 0=invalid
        return (m_u8.astype(np.uint8, copy=False) != 0).reshape(-1)

    def _mask_stats_for_action_id(self, action_id: int) -> tuple[bool, int, int]:
        mask = self._action_mask_bool()
        aid = int(action_id)
        masked = bool(aid < 0 or aid >= mask.size or (not bool(mask[aid])))
        masked_count = int((~mask).sum())
        return masked, masked_count, int(mask.size)

    def _episode_seed_from_np_random(self) -> int:
        return int(self.np_random.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))

    def _warmup_spec_for_reset(self) -> Any:
        if self._warmup is None:
            return None
        return self._warmup(game=self.game, rng=self.np_random)  # type: ignore[misc]

    def _snapshot(self) -> Dict[str, Any]:
        # Current binding supports snapshot(include_grid=True, visible=True)
        try:
            return cast(Dict[str, Any], self.game.snapshot(include_grid=True, visible=True))
        except TypeError:
            return cast(Dict[str, Any], self.game.snapshot(include_grid=True))

    def _step_features(self) -> Dict[str, Any]:
        # New binding: no visible arg (always visible grid)
        try:
            return cast(Dict[str, Any], self.game.step_features(prev=self._prev_feat))
        except TypeError:
            # Backward compat if older binding still has visible=
            return cast(Dict[str, Any], self.game.step_features(prev=self._prev_feat, visible=True))

    def _requested_rot_col_and_action_id(self, action: Any) -> tuple[int, int, int]:
        """
        Returns (requested_rot, requested_col, requested_action_id), using engine SSOT.
        """
        if self.action_mode == "discrete":
            requested_action_id = int(action)
            # SSOT decode (for logging/features)
            try:
                rot, col = self.game.decode_action_id(int(requested_action_id))
                return int(rot), int(col), int(requested_action_id)
            except Exception:
                # Out-of-range etc. -> keep sentinel rot/col
                return -1, -1, int(requested_action_id)

        # multidiscrete: action provides (rot, col)
        if isinstance(action, (tuple, list)) and len(action) == 2:
            rot, col = action
            requested_rot = int(rot)
            requested_col = int(col)
        else:
            arr = np.asarray(action).reshape(-1)
            if arr.size != 2:
                raise TypeError(f"invalid action for action_mode='multidiscrete': {action!r}")
            requested_rot = int(arr[0])
            requested_col = int(arr[1])

        # SSOT encode
        requested_action_id = int(self.game.encode_action_id(int(requested_rot), int(requested_col)))
        return int(requested_rot), int(requested_col), int(requested_action_id)

    # ---------------------------------------------------------------------
    # sb3-contrib MaskablePPO hook
    # ---------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
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

        self.game.reset(seed=int(ep_seed), warmup=warmup_spec)

        st = self._snapshot()
        self._last_state = st

        obs = self._obs_from_state(st)

        info = build_reset_info(
            state=st,
            episode_idx=int(self._episode_idx),
            episode_step=int(self._steps),
            action_mode=str(self.action_mode),
            piece_rule=self._piece_rule_name(),
        )
        info["invalid_action_policy"] = str(self.invalid_action_policy)
        info["warmup"] = None if self._warmup is None else getattr(self._warmup.__class__, "__name__", "warmup")
        info["episode_seed"] = int(ep_seed)
        return obs, info

    def step(self, action: Any):
        self._steps += 1
        prev_state = self._last_state
        if prev_state is None:
            raise RuntimeError("step() called before reset()")

        requested_rot, requested_col, requested_action_id = self._requested_rot_col_and_action_id(action)

        masked_action, masked_action_count, action_dim = self._mask_stats_for_action_id(requested_action_id)
        mask = self._action_mask_bool()
        requested_mask_invalid = bool(
            requested_action_id < 0
            or requested_action_id >= int(mask.size)
            or (not bool(mask[int(requested_action_id)]))
        )

        used_action_id = int(requested_action_id)

        terminated = False
        truncated = False
        cleared = 0

        # SSOT: invalid_action comes from engine if we stepped; otherwise policy.
        invalid_action = False

        if requested_mask_invalid:
            if self.invalid_action_policy == "terminate":
                terminated = True
                cleared = 0
                invalid_action = True
            else:
                # noop
                terminated = False
                cleared = 0
                invalid_action = True
        else:
            terminated, cleared, invalid_action_engine = self.game.step_action_id(int(used_action_id))
            invalid_action = bool(invalid_action_engine)

        # Mask mismatch debug (discrete only)
        mask_mismatch = False
        if self.action_mode == "discrete":
            mask_mismatch = bool(masked_action) != bool(invalid_action)

        if self.max_steps is not None and self._steps >= self.max_steps and not bool(terminated):
            truncated = True

        st = self._snapshot()
        self._last_state = st

        sf = self._step_features()
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

        prev_score = float(prev_state.get("score", 0.0))
        next_score = float(st.get("score", 0.0))
        delta_score = float(next_score - prev_score)

        # SSOT decode for used action id (if valid range)
        try:
            used_rot, used_col = self.game.decode_action_id(int(used_action_id))
            used_rot = int(used_rot)
            used_col = int(used_col)
        except Exception:
            used_rot, used_col = int(requested_rot), int(requested_col)

        placed_kind = str(prev_state.get("active_kind", "?"))

        # applied iff we actually stepped and engine said it was valid
        applied = bool((not requested_mask_invalid) and (not invalid_action))

        features = build_transition_features(
            cleared=int(cleared),
            terminated=bool(terminated),
            placed_kind=str(placed_kind),
            requested_rot=int(requested_rot),
            requested_col=int(requested_col),
            used_rot=int(used_rot),
            used_col=int(used_col),
            used_action_id=int(used_action_id),
            applied=bool(applied),
            invalid_action=bool(invalid_action),
            invalid_action_policy=None,
            masked_action=bool(masked_action),
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
            invalid_action=bool(invalid_action),
            delta_holes=delta_holes,
            delta_max_height=delta_max_height,
            delta_bumpiness=delta_bumpiness,
            bumpiness=bumpiness_after,
        )

        info = build_step_info_update(
            invalid_action=bool(invalid_action),
            invalid_action_policy=str(self.invalid_action_policy),
            remapped=False,
            applied=bool(applied),
            mask_mismatch=bool(mask_mismatch),
            game_over=bool(terminated),
            delta_score=float(delta_score),
            state=st,
            cleared=int(cleared),
            action_mode=str(self.action_mode),
            requested_rot=int(requested_rot),
            requested_col=int(requested_col),
            requested_action_id=int(requested_action_id),
            used_rot=int(used_rot),
            used_col=int(used_col),
            masked_action=bool(masked_action),
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
            engine_info={
            },
        )

        info_for_reward = dict(info)
        info_for_reward["engine_info"] = {"game": self.game}

        shaped = float(
            self.reward_fn(
                prev_state=prev_state,
                action=action,
                next_state=st,
                features=features,
                info=info_for_reward,
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
