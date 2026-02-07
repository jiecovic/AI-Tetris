# src/tetris_rl/core/training/evaluation/eval_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.training.metrics import StatsAccumulator, StatsAccumulatorConfig
from tetris_rl.core.training.env_factory import make_vec_env_from_cfg


def _obs_set(obs: Any, idx: int, value: Any) -> Any:
    """
    Set slot idx in a batched observation returned by VecEnv.

    Supports:
      - np.ndarray (B, ...)
      - dict[str, np.ndarray] where each array is (B, ...)
    """
    if isinstance(obs, dict) and isinstance(value, dict):
        out = dict(obs)
        for k, v in value.items():
            arr = out.get(k)
            if isinstance(arr, np.ndarray) and isinstance(v, np.ndarray):
                arr[idx] = v
                out[k] = arr
        return out

    if isinstance(obs, np.ndarray) and isinstance(value, np.ndarray):
        obs[idx] = value
        return obs

    return obs


def _vec_action_masks(vec_env: VecEnv, *, n_envs: int) -> Optional[np.ndarray]:
    try:
        masks_list = vec_env.env_method("action_masks")
    except Exception:
        return None

    if not isinstance(masks_list, list) or not masks_list:
        return None

    out: list[np.ndarray] = []
    for i in range(min(int(n_envs), len(masks_list))):
        m = masks_list[i]
        if m is None:
            return None
        try:
            arr = np.asarray(m, dtype=bool).reshape(-1)
        except Exception:
            return None
        out.append(arr)

    if len(out) != int(n_envs):
        return None

    action_dim = int(out[0].shape[0])
    for a in out[1:]:
        if int(a.shape[0]) != action_dim:
            return None

    try:
        return np.stack(out, axis=0)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _effective_algo_type_from_model(model: Any) -> str:
    t = getattr(model, "_tetris_algo_type", None)
    if isinstance(t, str) and t.strip():
        s = t.strip().lower()
        if s in {"ppo", "maskable_ppo"}:
            return s

    name = model.__class__.__name__.strip().lower()
    return "maskable_ppo" if "maskable" in name else "ppo"


@dataclass
class _SlotState:
    seed: int
    ep_reward: float = 0.0
    ep_steps: int = 0


def _build_eval_vec_env(
    *,
    cfg: Dict[str, Any],
    run_cfg: RunConfig,
) -> Tuple[VecEnv, Any]:
    """
    Build a fresh eval VecEnv.

    Semantics:
      - uses run.n_envs (single knob)
      - forces vec="dummy"
    """
    eval_run: RunConfig = run_cfg.model_copy(update={"vec": "dummy"})

    built = make_vec_env_from_cfg(cfg=cfg, run_cfg=eval_run)
    return built.vec_env, built  # keep built alive for any held refs


def evaluate_model(
        *,
        model: Any,
        cfg: Dict[str, Any],
        run_cfg: RunConfig,
        eval_steps: int,
        deterministic: bool,
        seed_base: int,
        num_envs: int = 1,
        on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
        on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Step-budget evaluation (canonical).

    Runs until we have collected >= eval_steps total env steps across all VecEnv slots.
    Aggregates per-step metric contract from info["tf"]/info["game"] via StatsAccumulator.

    Episode-level metrics are computed only over episodes that happen to complete within the step budget.
    """
    eval_steps = int(eval_steps)
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be > 0, got {eval_steps}")

    # Single knob: run.n_envs controls eval vec env size (eval.num_envs is ignored here).
    _ = num_envs

    algo_type = _effective_algo_type_from_model(model)
    want_masking = algo_type == "maskable_ppo"

    vec_env, _built = _build_eval_vec_env(cfg=cfg, run_cfg=run_cfg)
    obs = vec_env.reset()

    # Infer n_envs from the built VecEnv.
    try:
        n_envs = int(getattr(vec_env, "num_envs", 1))
    except Exception:
        n_envs = 1
    n_envs = max(1, int(n_envs))

    # Seed each slot explicitly (SB3-compatible). Use seed_base+i for the initial episodes.
    slots: list[_SlotState] = []
    for i in range(n_envs):
        s = _SlotState(seed=int(seed_base) + i)
        slots.append(s)

        ret = vec_env.env_method("reset", seed=int(s.seed), indices=i)

        obs_i = None
        try:
            x = ret[0]
            if isinstance(x, tuple) and len(x) == 2:
                obs_i = x[0]
            else:
                obs_i = x
        except Exception:
            obs_i = ret[0] if ret else None

        if obs_i is not None:
            obs = _obs_set(obs, i, obs_i)

    acc = StatsAccumulator(cfg=StatsAccumulatorConfig(log_action_histograms=False))

    completed_episodes = 0
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    ep_final_scores: list[Optional[float]] = []
    ep_final_lines: list[Optional[float]] = []
    ep_max_levels: list[Optional[float]] = []

    next_seed = int(seed_base) + n_envs
    warned_no_masks = False

    try:
        while acc.steps < eval_steps:
            masks: Optional[np.ndarray] = None
            if want_masking:
                masks = _vec_action_masks(vec_env, n_envs=n_envs)
                if masks is not None:
                    actions, _ = model.predict(obs, deterministic=bool(deterministic), action_masks=masks)
                else:
                    if not warned_no_masks:
                        warned_no_masks = True
                        print(
                            "[eval] WARN: maskable_ppo model but eval env did not provide action masks.\n"
                            "[eval] WARN: falling back to unmasked predict()."
                        )
                    actions, _ = model.predict(obs, deterministic=bool(deterministic))
            else:
                actions, _ = model.predict(obs, deterministic=bool(deterministic))

            obs, step_rewards, dones, infos = vec_env.step(actions)

            step_rewards = np.asarray(step_rewards, dtype=np.float64).reshape((n_envs,))
            dones = np.asarray(dones, dtype=bool).reshape((n_envs,))
            infos_list = cast(list[dict], infos)

            for i in range(n_envs):
                if acc.steps >= eval_steps:
                    break

                info_i = infos_list[i] if i < len(infos_list) and isinstance(infos_list[i], dict) else {}
                acc.ingest_info(info_i)
                if on_step is not None:
                    on_step(1)

                # Secondary episode stats
                slots[i].ep_reward += float(step_rewards[i])
                slots[i].ep_steps += 1

                if not dones[i]:
                    continue

                completed_episodes += 1
                ep_returns.append(float(slots[i].ep_reward))
                ep_steps.append(int(slots[i].ep_steps))

                game = info_i.get("game") if isinstance(info_i, dict) else None
                if isinstance(game, Mapping):
                    ep_final_scores.append(_as_float(game.get("score")))
                    ep_final_lines.append(_as_float(game.get("lines_total")))
                    ep_max_levels.append(_as_float(game.get("level")))
                else:
                    ep_final_scores.append(None)
                    ep_final_lines.append(None)
                    ep_max_levels.append(None)

                if on_episode is not None:
                    on_episode(completed_episodes, float(ep_returns[-1]))

                # Reset this slot and reseed deterministically
                new_seed = int(next_seed)
                next_seed += 1

                ret = vec_env.env_method("reset", seed=int(new_seed), indices=i)

                obs_i = None
                try:
                    x = ret[0]
                    if isinstance(x, tuple) and len(x) == 2:
                        obs_i = x[0]
                    else:
                        obs_i = x
                except Exception:
                    obs_i = ret[0] if ret else None

                if obs_i is not None:
                    obs = _obs_set(obs, i, obs_i)

                slots[i] = _SlotState(seed=int(new_seed))
    finally:
        vec_env.close()

    out: Dict[str, Any] = {}
    out.update(acc.summarize())

    out["eval/steps"] = int(acc.steps)
    out["eval/deterministic"] = bool(deterministic)
    out["eval/seed_base"] = int(seed_base)
    out["eval/num_envs"] = int(n_envs)
    out["eval/algo_type"] = str(algo_type)

    out["episode/completed_episodes"] = int(completed_episodes)

    def _mean(xs: Sequence[float]) -> Optional[float]:
        if not xs:
            return None
        return float(np.asarray(xs, dtype=np.float64).mean())

    def _mean_opt(xs: Sequence[Optional[float]]) -> Optional[float]:
        vals = [float(v) for v in xs if v is not None]
        if not vals:
            return None
        return float(np.asarray(vals, dtype=np.float64).mean())

    m = _mean(ep_returns)
    if m is not None:
        out["episode/return_mean"] = float(m)
    m = _mean([float(x) for x in ep_steps])
    if m is not None:
        out["episode/steps_mean"] = float(m)
    m = _mean_opt(ep_final_lines)
    if m is not None:
        out["episode/lines_mean"] = float(m)
    m = _mean_opt(ep_final_scores)
    if m is not None:
        out["episode/final_score_mean"] = float(m)
    m = _mean_opt(ep_max_levels)
    if m is not None:
        out["episode/max_level_mean"] = float(m)

    # purely descriptive (non-semantic) metadata from cfg is allowed until Run/Env specs exist
    try:
        env = cfg.get("env", {}) if isinstance(cfg, dict) else {}
        if not isinstance(env, dict):
            env = {}
        out["meta/env_type"] = str(env.get("type", ""))
        obs_cfg = env.get("obs", {}) or {}
        out["meta/obs_type"] = str(obs_cfg.get("type", "")) if isinstance(obs_cfg, dict) else ""
    except Exception:
        pass

    return out


__all__ = ["evaluate_model"]
