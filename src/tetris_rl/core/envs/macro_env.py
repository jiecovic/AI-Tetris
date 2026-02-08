# src/tetris_rl/core/envs/macro_env.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris_rl.core.envs.actions import action_mask_bool, resolve_action_request
from tetris_rl.core.envs.api import RewardFn
from tetris_rl.core.envs.config import MacroEnvParams
from tetris_rl.core.envs.macro_info import build_reset_info
from tetris_rl.core.envs.obs.macro_state import (
    MacroObsSpec,
    build_macro_obs_space,
    encode_macro_obs,
)
from tetris_rl.core.envs.stepper import MacroStepRunner


class MacroTetrisEnv(gym.Env):
    """
    Macro placement environment backed by the Rust engine (PyO3).

    Authoritative sources:
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
            spec: MacroEnvParams,
    ) -> None:
        super().__init__()
        self.game = game
        self.reward_fn = reward_fn
        self.max_steps = int(spec.max_steps) if spec.max_steps is not None else None
        self.action_mode = spec.action_mode

        self.invalid_action_policy = str(spec.invalid_action_policy).strip().lower()
        if self.invalid_action_policy not in {"noop", "terminate"}:
            raise ValueError(f"unknown invalid_action_policy: {spec.invalid_action_policy!r}")


        # Geometry from engine (authoritative).
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
        self._stepper = MacroStepRunner(self)

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

    def _episode_seed_from_np_random(self) -> int:
        return int(self.np_random.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))

    def value_features(
        self,
        *,
        features: list[str],
        action_id: int | None = None,
        after_clear: bool = True,
        visible: bool = False,
    ) -> list[float]:
        if action_id is None:
            vals = self.game.heuristic_features(list(features), False, bool(visible))
            return [float(v) for v in vals]
        vals = self.game.simulate_active_features(
            int(action_id),
            list(features),
            bool(after_clear),
            False,
            bool(visible),
        )
        if vals is None:
            raise RuntimeError("simulate_active_features returned None for the requested action_id")
        return [float(v) for v in vals]

    def value_features_for_action(
        self,
        *,
        features: list[str],
        action: Any,
        after_clear: bool = True,
        visible: bool = False,
    ) -> list[float]:
        req = resolve_action_request(action=action, action_mode=str(self.action_mode), game=self.game)
        action_id = int(req.requested_action_id)
        vals = self.game.simulate_active_features(
            int(action_id),
            list(features),
            bool(after_clear),
            False,
            bool(visible),
        )
        if vals is None:
            raise RuntimeError("simulate_active_features returned None for the requested action")
        return [float(v) for v in vals]


    def _snapshot(self) -> Dict[str, Any]:
        # Current binding supports snapshot(include_grid=True, visible=True)
        try:
            return cast(Dict[str, Any], self.game.snapshot(include_grid=True, visible=True))
        except TypeError:
            return cast(Dict[str, Any], self.game.snapshot(include_grid=True))

    def _step_features(self) -> Dict[str, Any]:
        # New binding: no visible arg (always visible grid)
        return cast(Dict[str, Any], self.game.step_features(prev=self._prev_feat))

    # ---------------------------------------------------------------------
    # sb3-contrib MaskablePPO hook
    # ---------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        return action_mask_bool(self.game)

    # ---------------------------------------------------------------------
    # gym API
    # ---------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._episode_idx += 1
        self._prev_feat = None

        ep_seed = int(seed) if seed is not None else self._episode_seed_from_np_random()
        self.game.reset(seed=int(ep_seed))

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
        info["warmup"] = None
        info["episode_seed"] = int(ep_seed)
        return obs, info

    def step(self, action: Any):
        return self._stepper.step(action)

    @property
    def last_state(self) -> Dict[str, Any]:
        st = self._last_state
        if st is None:
            raise RuntimeError("MacroTetrisEnv.last_state accessed before reset()")
        return st
