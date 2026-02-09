# src/tetris_rl/core/training/evaluation/eval_runner.py
from __future__ import annotations

import signal
import tempfile
import threading
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from planning_rl.utils.seed import seed32_from
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.training.config import AlgoConfig, ImitationAlgoParams
from tetris_rl.core.training.env_factory import make_vec_env_from_cfg
from tetris_rl.core.training.imitation.algorithm import ImitationAlgorithm
from tetris_rl.core.training.metrics import StatsAccumulator, StatsAccumulatorConfig
from tetris_rl.core.training.model_io import load_model_from_algo_config


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
    num_envs: int,
) -> Tuple[VecEnv, Any]:
    """
    Build a fresh eval VecEnv.

    Semantics:
      - uses num_envs for eval parallelism
      - forces vec="dummy"
    """
    eval_run: RunConfig = run_cfg.model_copy(
        update={"vec": "dummy", "n_envs": int(num_envs), "max_episode_steps": None},
    )

    built = make_vec_env_from_cfg(cfg=cfg, run_cfg=eval_run)
    return built.vec_env, built  # keep built alive for any held refs


def _eval_state_loop(
    *,
    model: Any,
    vec_env: VecEnv,
    eval_episodes: int,
    min_steps: int,
    max_steps_per_episode: Optional[int],
    deterministic: bool,
    seed_base: int,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    eval_episodes = int(eval_episodes)
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_episodes}")
    min_steps = int(min_steps)
    if min_steps < 0:
        raise ValueError(f"min_steps must be >= 0, got {min_steps}")

    algo_type = _effective_algo_type_from_model(model)
    want_masking = algo_type == "maskable_ppo"

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
    total_reward = 0.0
    total_steps = 0

    next_seed = int(seed_base) + n_envs
    warned_no_masks = False
    active = [True for _ in range(int(n_envs))]
    draining = False

    max_steps_per_episode = None if max_steps_per_episode is None else int(max_steps_per_episode)

    while True:
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

        if not all(active):
            try:
                actions_arr = np.asarray(actions)
                inactive = [i for i, a in enumerate(active) if not a]
                if actions_arr.ndim == 1:
                    actions_arr[inactive] = 0
                else:
                    actions_arr[inactive] = 0
                actions = actions_arr
            except Exception:
                pass

        obs, step_rewards, dones, infos = vec_env.step(actions)

        step_rewards = np.asarray(step_rewards, dtype=np.float64).reshape((n_envs,))
        dones = np.asarray(dones, dtype=bool).reshape((n_envs,))
        infos_list = cast(list[dict], infos)

        stop_after_batch = False
        for i in range(n_envs):
            if not active[i]:
                continue
            info_i = infos_list[i] if i < len(infos_list) and isinstance(infos_list[i], dict) else {}
            acc.ingest_info(info_i)
            if on_step is not None:
                on_step(1)

            # Secondary episode stats
            slots[i].ep_reward += float(step_rewards[i])
            slots[i].ep_steps += 1
            total_reward += float(step_rewards[i])
            total_steps += 1

            done = bool(dones[i])
            cap_reached = max_steps_per_episode is not None and int(slots[i].ep_steps) >= int(max_steps_per_episode)
            if cap_reached and not done:
                done = True

            if not done:
                continue

            completed_episodes += 1
            ep_returns.append(float(slots[i].ep_reward))
            ep_steps.append(int(slots[i].ep_steps))

            game = info_i.get("game") if isinstance(info_i, dict) else None
            if isinstance(game, Mapping):
                ep_final_scores.append(_as_float(game.get("score")))
                ep_final_lines.append(_as_float(game.get("lines_total")))
            else:
                ep_final_scores.append(None)
                ep_final_lines.append(None)

            if on_episode is not None:
                on_episode(completed_episodes, float(ep_returns[-1]))

            if not draining and completed_episodes >= eval_episodes and int(total_steps) >= int(min_steps):
                draining = True

            # Reset this slot and reseed deterministically
            if draining:
                active[i] = False
                stop_after_batch = True
                continue

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

        if stop_after_batch and draining and not any(active):
            break

    return {
        "acc_state": acc.to_state(),
        "completed_episodes": int(completed_episodes),
        "ep_returns": list(ep_returns),
        "ep_steps": list(ep_steps),
        "ep_final_scores": list(ep_final_scores),
        "ep_final_lines": list(ep_final_lines),
        "cur_ep_steps": [int(s.ep_steps) for s in slots if int(s.ep_steps) > 0],
        "n_envs": int(n_envs),
        "algo_type": str(algo_type),
        "total_reward": float(total_reward),
        "total_steps": int(total_steps),
        "eval_episodes_target": int(eval_episodes),
        "min_steps": int(min_steps),
        "max_steps_per_episode": None if max_steps_per_episode is None else int(max_steps_per_episode),
    }


def _eval_state(
    *,
    model: Any,
    cfg: Dict[str, Any],
    run_cfg: RunConfig,
    eval_episodes: int,
    min_steps: int,
    max_steps_per_episode: Optional[int],
    deterministic: bool,
    seed_base: int,
    num_envs: int,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    vec_env, _built = _build_eval_vec_env(cfg=cfg, run_cfg=run_cfg, num_envs=num_envs)
    try:
        return _eval_state_loop(
            model=model,
            vec_env=vec_env,
            eval_episodes=eval_episodes,
            min_steps=min_steps,
            max_steps_per_episode=max_steps_per_episode,
            deterministic=deterministic,
            seed_base=seed_base,
            on_episode=on_episode,
            on_step=on_step,
        )
    finally:
        vec_env.close()


def _summarize_eval_states(
    *,
    states: Sequence[Mapping[str, Any]],
    eval_episodes: int,
    min_steps: int,
    max_steps_per_episode: Optional[int],
    seed_base: int,
    deterministic: bool,
    num_envs: int,
    algo_type: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    acc = StatsAccumulator(cfg=StatsAccumulatorConfig(log_action_histograms=False))
    completed_episodes = 0
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    ep_final_scores: list[Optional[float]] = []
    ep_final_lines: list[Optional[float]] = []
    cur_steps: list[int] = []
    total_reward = 0.0
    total_steps = 0

    for state in states:
        acc_state = state.get("acc_state", {})
        if isinstance(acc_state, Mapping):
            acc.merge_state(acc_state)
        completed_episodes += int(state.get("completed_episodes", 0))
        ep_returns.extend(state.get("ep_returns", []) or [])
        ep_steps.extend(state.get("ep_steps", []) or [])
        ep_final_scores.extend(state.get("ep_final_scores", []) or [])
        ep_final_lines.extend(state.get("ep_final_lines", []) or [])
        cur_steps.extend(state.get("cur_ep_steps", []) or [])
        total_reward += float(state.get("total_reward", 0.0))
        total_steps += int(state.get("total_steps", 0))

    out: Dict[str, Any] = {}
    out.update(acc.summarize())

    out["eval/steps"] = int(acc.steps)
    out["eval/episodes_target"] = int(eval_episodes)
    out["eval/min_steps"] = int(min_steps)
    if max_steps_per_episode is not None:
        out["eval/max_steps_per_episode"] = int(max_steps_per_episode)
    out["eval/deterministic"] = bool(deterministic)
    out["eval/seed_base"] = int(seed_base)
    out["eval/num_envs"] = int(num_envs)
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
        out["episode/steps_completed_mean"] = float(m)

    steps_all = [float(x) for x in ep_steps] + [float(x) for x in cur_steps if int(x) > 0]
    m = _mean(steps_all)
    if m is not None:
        out["episode/steps_mean"] = float(m)
    m = _mean_opt(ep_final_lines)
    if m is not None:
        out["episode/lines_mean"] = float(m)
    m = _mean_opt(ep_final_scores)
    if m is not None:
        out["episode/final_score_mean"] = float(m)

    if total_steps > 0:
        out["episode/return_per_step"] = float(total_reward) / float(total_steps)

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


def evaluate_model(
    *,
    model: Any,
    cfg: Dict[str, Any],
    run_cfg: RunConfig,
    eval_episodes: int,
    min_steps: int,
    max_steps_per_episode: Optional[int],
    deterministic: bool,
    seed_base: int,
    num_envs: int = 1,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Episode-target evaluation (canonical).

    Runs until we have collected >= eval_episodes completed episodes and >= min_steps total env steps.
    Aggregates per-step metric contract from info["tf"]/info["game"] via StatsAccumulator.

    Episode-level metrics are computed only over completed episodes.
    """
    eval_episodes = int(eval_episodes)
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_episodes}")
    min_steps = int(min_steps)
    if min_steps < 0:
        raise ValueError(f"min_steps must be >= 0, got {min_steps}")

    num_envs = max(1, int(num_envs))
    algo_type = _effective_algo_type_from_model(model)

    state = _eval_state(
        model=model,
        cfg=cfg,
        run_cfg=run_cfg,
        eval_episodes=eval_episodes,
        min_steps=min_steps,
        max_steps_per_episode=max_steps_per_episode,
        deterministic=deterministic,
        seed_base=seed_base,
        num_envs=num_envs,
        on_episode=on_episode,
        on_step=on_step,
    )

    return _summarize_eval_states(
        states=[state],
        eval_episodes=eval_episodes,
        min_steps=min_steps,
        max_steps_per_episode=max_steps_per_episode,
        seed_base=seed_base,
        deterministic=deterministic,
        num_envs=int(state.get("n_envs", num_envs)),
        algo_type=str(algo_type),
        cfg=cfg,
    )


_WORKER_CFG: Dict[str, Any] | None = None
_WORKER_RUN_CFG: Dict[str, Any] | None = None
_WORKER_MODEL_PATH: str | None = None
_WORKER_ALGO_BLOCK: Dict[str, Any] | None = None
_WORKER_DEVICE: str = "cpu"
_WORKER_PROGRESS: Any | None = None


def _init_eval_worker(
    cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    model_path: str,
    algo_block: Dict[str, Any],
    device: str,
    progress_queue: Any | None,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_CFG, _WORKER_RUN_CFG, _WORKER_MODEL_PATH, _WORKER_ALGO_BLOCK, _WORKER_DEVICE, _WORKER_PROGRESS
    _WORKER_CFG = cfg
    _WORKER_RUN_CFG = run_cfg
    _WORKER_MODEL_PATH = model_path
    _WORKER_ALGO_BLOCK = algo_block
    _WORKER_DEVICE = device
    _WORKER_PROGRESS = progress_queue


def _load_worker_model(*, env: Any) -> Any:
    if _WORKER_MODEL_PATH is None or _WORKER_ALGO_BLOCK is None:
        raise RuntimeError("worker not initialized")

    algo_type = str(_WORKER_ALGO_BLOCK.get("type", "")).strip().lower()
    if algo_type == "imitation":
        params = ImitationAlgoParams.model_validate(_WORKER_ALGO_BLOCK.get("params", {}) or {})
        return ImitationAlgorithm.load(
            _WORKER_MODEL_PATH,
            env=env,
            params=params,
            device=str(_WORKER_DEVICE),
        )

    algo_cfg = AlgoConfig.model_validate(_WORKER_ALGO_BLOCK)
    loaded = load_model_from_algo_config(
        algo_cfg=algo_cfg,
        ckpt=Path(_WORKER_MODEL_PATH),
        device=str(_WORKER_DEVICE),
        env=env,
    )
    return loaded.model


def _worker_eval(args: tuple[int, int, int, int, Optional[int]]) -> Dict[str, Any]:
    eval_episodes, min_steps, seed_base, deterministic, max_steps_per_episode = args
    if _WORKER_CFG is None or _WORKER_RUN_CFG is None:
        raise RuntimeError("worker not initialized")
    steps_buf = 0

    def _on_step(k: int) -> None:
        nonlocal steps_buf
        steps_buf += int(k)

    def _on_episode(_done_eps: int, ret: Optional[float]) -> None:
        nonlocal steps_buf
        if _WORKER_PROGRESS is not None:
            _WORKER_PROGRESS.put(("episode", int(steps_buf), _as_float(ret)))
        steps_buf = 0

    run_cfg = RunConfig.model_validate(_WORKER_RUN_CFG)
    vec_env, _built = _build_eval_vec_env(cfg=_WORKER_CFG, run_cfg=run_cfg, num_envs=1)
    try:
        model = _load_worker_model(env=vec_env)
        on_episode = _on_episode if _WORKER_PROGRESS is not None else None
        on_step = _on_step if _WORKER_PROGRESS is not None else None
        return _eval_state_loop(
            model=model,
            vec_env=vec_env,
            eval_episodes=int(eval_episodes),
            min_steps=int(min_steps),
            max_steps_per_episode=max_steps_per_episode,
            deterministic=bool(deterministic),
            seed_base=int(seed_base),
            on_episode=on_episode,
            on_step=on_step,
        )
    finally:
        vec_env.close()


def evaluate_model_workers(
    *,
    model: Any,
    cfg: Dict[str, Any],
    run_cfg: RunConfig,
    eval_episodes: int,
    min_steps: int,
    max_steps_per_episode: Optional[int],
    deterministic: bool,
    seed_base: int,
    workers: int,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    workers = max(1, int(workers))
    eval_episodes = int(eval_episodes)
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_episodes}")
    min_steps = int(min_steps)
    if min_steps < 0:
        raise ValueError(f"min_steps must be >= 0, got {min_steps}")

    if workers <= 1:
        return evaluate_model(
            model=model,
            cfg=cfg,
            run_cfg=run_cfg,
            eval_episodes=eval_episodes,
            min_steps=min_steps,
            max_steps_per_episode=max_steps_per_episode,
            deterministic=deterministic,
            seed_base=seed_base,
            num_envs=1,
            on_episode=on_episode,
            on_step=on_step,
        )

    algo_type = _effective_algo_type_from_model(model)

    workers = min(int(workers), int(eval_episodes)) if int(eval_episodes) > 0 else 1
    per_worker_eps = eval_episodes // workers
    eps_remainder = eval_episodes % workers
    per_worker_steps = min_steps // workers if workers > 0 else 0
    steps_remainder = min_steps % workers if workers > 0 else 0
    tasks: list[tuple[int, int, int, int, Optional[int]]] = []
    for i in range(int(workers)):
        eps_i = int(per_worker_eps + (1 if i < eps_remainder else 0))
        steps_i = int(per_worker_steps + (1 if i < steps_remainder else 0))
        if eps_i <= 0:
            continue
        seed_i = int(seed32_from(base_seed=int(seed_base), stream_id=int(0xE9A1 + i)))
        tasks.append((eps_i, steps_i, seed_i, int(deterministic), max_steps_per_episode))

    if not tasks:
        raise ValueError("no eval tasks to run")

    algo_block = cfg.get("algo", {}) if isinstance(cfg, dict) else {}
    if not isinstance(algo_block, dict):
        raise TypeError("algo config must be a mapping")

    with tempfile.TemporaryDirectory(prefix="tetris_eval_") as tmpdir:
        tmp_path = Path(tmpdir) / "eval_model.zip"
        model.save(str(tmp_path))

        ctx = get_context("spawn")
        states: list[Dict[str, Any]] = []
        pool = None
        terminated = False
        prev_handler = signal.getsignal(signal.SIGINT)
        progress_queue = None
        progress_thread = None
        done_eps = 0

        def _sigint_handler(_signum, _frame) -> None:
            if pool is not None:
                pool.terminate()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            if on_episode is not None or on_step is not None:
                progress_queue = ctx.Queue()

                def _drain_progress() -> None:
                    nonlocal done_eps
                    if progress_queue is None:
                        return
                    while True:
                        msg = progress_queue.get()
                        if msg is None:
                            break
                        kind, steps, ret = msg
                        if kind != "episode":
                            continue
                        if on_step is not None and int(steps) > 0:
                            on_step(int(steps))
                        if on_episode is not None:
                            done_eps += 1
                            on_episode(int(done_eps), _as_float(ret))

                progress_thread = threading.Thread(target=_drain_progress, daemon=True)
                progress_thread.start()

            pool = ctx.Pool(
                processes=len(tasks),
                initializer=_init_eval_worker,
                initargs=(
                    dict(cfg),
                    run_cfg.model_dump(),
                    str(tmp_path),
                    dict(algo_block),
                    str(run_cfg.device),
                    progress_queue,
                ),
            )
            for state in pool.imap(_worker_eval, tasks):
                states.append(state)
                if progress_queue is None and on_step is not None:
                    acc_state = state.get("acc_state", {})
                    steps = int(acc_state.get("n_steps", 0)) if isinstance(acc_state, Mapping) else 0
                    if steps > 0:
                        on_step(int(steps))
        except KeyboardInterrupt:
            terminated = True
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if pool is not None and not terminated:
                pool.close()
                pool.join()
            if progress_queue is not None:
                progress_queue.put(None)
            if progress_thread is not None:
                progress_thread.join()

    return _summarize_eval_states(
        states=states,
        eval_episodes=eval_episodes,
        min_steps=min_steps,
        max_steps_per_episode=max_steps_per_episode,
        seed_base=seed_base,
        deterministic=deterministic,
        num_envs=len(tasks),
        algo_type=str(algo_type),
        cfg=cfg,
    )


__all__ = ["evaluate_model", "evaluate_model_workers"]
