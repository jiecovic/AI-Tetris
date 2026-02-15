# src/tetris_rl/core/envs/stepper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

from tetris_rl.core.envs.actions import (
    action_mask_bool,
    mask_stats_for_action_id,
    resolve_action_request,
)
from tetris_rl.core.envs.macro_info import build_step_payload_for_env


@dataclass(frozen=True)
class EngineStep:
    terminated: bool
    cleared: int
    invalid_action: bool
    sf: Dict[str, Any]


class MacroStepRunner:
    def __init__(self, env: Any) -> None:
        self.env = env

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        env = self.env
        env._steps += 1

        prev_state = env._last_state
        if prev_state is None:
            raise RuntimeError("step() called before reset()")

        # ------------------------------------------------------------
        # 1) Resolve requested action + mask stats
        # ------------------------------------------------------------
        req = resolve_action_request(action=action, action_mode=str(env.action_mode), game=env.game)
        mask_stats = mask_stats_for_action_id(
            action_id=req.requested_action_id,
            mask=action_mask_bool(env.game),
        )

        # ------------------------------------------------------------
        # 2) Apply action (or noop/terminate) and collect step features
        # ------------------------------------------------------------
        step_out = self._apply_action(req, mask_stats)

        # ------------------------------------------------------------
        # 3) Truncation and next-state snapshot
        # ------------------------------------------------------------
        truncated = False
        if env.max_steps is not None and env._steps >= env.max_steps and not bool(step_out.terminated):
            truncated = True

        st = env._snapshot()
        env._last_state = st

        # ------------------------------------------------------------
        # 4) Decode used action identity (for logging)
        # ------------------------------------------------------------
        applied = bool((not mask_stats.masked_action) and (not step_out.invalid_action))
        used_rot, used_col = self._decode_used_action(
            applied=applied,
            requested=req,
            used_action_id=int(req.requested_action_id),
        )

        # ------------------------------------------------------------
        # 5) Build payload (features + info)
        # ------------------------------------------------------------
        payload = build_step_payload_for_env(
            env=env,
            sf=step_out.sf,
            requested=req,
            mask_stats=mask_stats,
            used_action_id=int(req.requested_action_id),
            used_rot=int(used_rot),
            used_col=int(used_col),
            invalid_action=bool(step_out.invalid_action),
            terminated=bool(step_out.terminated),
            cleared=int(step_out.cleared),
            prev_state=prev_state,
            next_state=st,
        )
        if payload.prev_feat is not None:
            env._prev_feat = payload.prev_feat

        # ------------------------------------------------------------
        # 6) Reward and return
        # ------------------------------------------------------------
        shaped = self._compute_reward(
            prev_state=prev_state,
            next_state=st,
            action=action,
            payload=payload,
        )

        if str(getattr(env, "info_level", "train")).strip().lower() != "watch":
            payload.info.pop("ui", None)

        obs = env._obs_from_state(st)
        return obs, shaped, bool(step_out.terminated), bool(truncated), payload.info

    def _apply_action(self, req, mask_stats) -> EngineStep:
        env = self.env
        if mask_stats.masked_action:
            terminated = bool(env.invalid_action_policy == "terminate")
            cleared = 0
            sf = env._step_features()
            return EngineStep(terminated=terminated, cleared=int(cleared), invalid_action=True, sf=sf)

        if not hasattr(env.game, "step_action_id_with_features"):
            raise RuntimeError("engine missing step_action_id_with_features; rebuild the Rust extension")

        terminated, cleared, invalid_action_engine, sf_obj = env.game.step_action_id_with_features(
            int(req.requested_action_id),
            env._prev_feat,
            bool(getattr(env, "_after_clear_features", lambda: True)()),
        )
        return EngineStep(
            terminated=bool(terminated),
            cleared=int(cleared),
            invalid_action=bool(invalid_action_engine),
            sf=cast(Dict[str, Any], sf_obj),
        )

    def _decode_used_action(self, *, applied: bool, requested, used_action_id: int) -> Tuple[int, int]:
        env = self.env
        if not applied:
            return int(requested.requested_rot), int(requested.requested_col)
        try:
            rot, col = env.game.decode_action_id(int(used_action_id))
            return int(rot), int(col)
        except Exception as exc:
            raise RuntimeError(f"engine failed to decode action_id={int(used_action_id)}") from exc

    def _compute_reward(
        self,
        *,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any],
        action: Any,
        payload,
    ) -> float:
        env = self.env
        info_for_reward = dict(payload.info)
        info_for_reward["engine_info"] = {"game": env.game}
        return float(
            env.reward_fn(
                prev_state=prev_state,
                action=action,
                next_state=next_state,
                features=payload.features,
                info=info_for_reward,
            )
        )


__all__ = ["MacroStepRunner", "EngineStep"]
