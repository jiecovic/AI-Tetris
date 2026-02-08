# src/tetris_rl/core/training/evaluation/eval_planning.py
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from multiprocessing import get_context
import signal

import numpy as np

from tetris_rl.core.training.metrics import StatsAccumulator, StatsAccumulatorConfig
from planning_rl.utils.seed import seed32_from
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.policies.planning_policies.heuristic_policy import HeuristicPlanningPolicy
from tetris_rl.core.policies.spec import HeuristicSpec


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _episode_seed(*, base_seed: int, episode_idx: int) -> int:
    return int(seed32_from(base_seed=int(base_seed), stream_id=int(episode_idx)))


def _planning_eval_state(
    *,
    policy: Any,
    env: Any,
    eval_steps: int,
    seed_base: int,
    deterministic: bool,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    eval_steps = int(eval_steps)
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be > 0, got {eval_steps}")

    _ = deterministic

    acc = StatsAccumulator(cfg=StatsAccumulatorConfig(log_action_histograms=False))

    completed_episodes = 0
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    ep_final_scores: list[Optional[float]] = []
    ep_final_lines: list[Optional[float]] = []
    total_reward = 0.0
    total_steps = 0

    episode_idx = 0
    next_seed = _episode_seed(base_seed=int(seed_base), episode_idx=episode_idx)
    episode_idx += 1
    obs, info = env.reset(seed=int(next_seed))
    _ = obs
    _ = info

    cur_ep_reward = 0.0
    cur_ep_steps = 0

    while acc.steps < eval_steps:
        action = policy.predict(env=env)
        obs, reward, terminated, truncated, info = env.step(action)
        _ = obs

        acc.ingest_info(info if isinstance(info, Mapping) else {})
        if on_step is not None:
            on_step(1)

        cur_ep_reward += float(reward)
        cur_ep_steps += 1
        total_reward += float(reward)
        total_steps += 1

        if not (terminated or truncated):
            continue

        completed_episodes += 1
        ep_returns.append(float(cur_ep_reward))
        ep_steps.append(int(cur_ep_steps))

        game = info.get("game") if isinstance(info, Mapping) else None
        if isinstance(game, Mapping):
            ep_final_scores.append(_as_float(game.get("score")))
            ep_final_lines.append(_as_float(game.get("lines_total")))
        else:
            ep_final_scores.append(None)
            ep_final_lines.append(None)

        if on_episode is not None:
            on_episode(completed_episodes, float(ep_returns[-1]))

        cur_ep_reward = 0.0
        cur_ep_steps = 0

        next_seed = _episode_seed(base_seed=int(seed_base), episode_idx=episode_idx)
        episode_idx += 1
        _obs2, _info2 = env.reset(seed=int(next_seed))
        _ = _obs2
        _ = _info2

    return {
        "acc_state": acc.to_state(),
        "completed_episodes": int(completed_episodes),
        "ep_returns": list(ep_returns),
        "ep_steps": list(ep_steps),
        "ep_final_scores": list(ep_final_scores),
        "ep_final_lines": list(ep_final_lines),
        "cur_ep_steps": int(cur_ep_steps),
        "total_reward": float(total_reward),
        "total_steps": int(total_steps),
    }


def _summarize_eval_state(
    *,
    states: Sequence[Mapping[str, Any]],
    eval_steps: int,
    seed_base: int,
    deterministic: bool,
    num_envs: int,
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
        cur_steps.append(int(state.get("cur_ep_steps", 0)))
        total_reward += float(state.get("total_reward", 0.0))
        total_steps += int(state.get("total_steps", 0))

    out: Dict[str, Any] = {}
    out.update(acc.summarize())

    out["eval/steps"] = int(acc.steps)
    out["eval/deterministic"] = bool(deterministic)
    out["eval/seed_base"] = int(seed_base)
    out["eval/num_envs"] = int(num_envs)
    out["eval/algo_type"] = "planning"

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

    steps_all = [float(x) for x in ep_steps]
    steps_all.extend([float(x) for x in cur_steps if int(x) > 0])
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

    return out


def evaluate_planning_policy(
    *,
    policy: Any,
    env: Any,
    eval_steps: int,
    seed_base: int,
    deterministic: bool,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Step-budget evaluation for planning-style policies (env-driven predict).
    """
    state = _planning_eval_state(
        policy=policy,
        env=env,
        eval_steps=eval_steps,
        seed_base=seed_base,
        deterministic=deterministic,
        on_episode=on_episode,
        on_step=on_step,
    )
    return _summarize_eval_state(
        states=[state],
        eval_steps=eval_steps,
        seed_base=seed_base,
        deterministic=deterministic,
        num_envs=1,
    )


_WORKER_ENV: Any | None = None
_WORKER_POLICY: HeuristicPlanningPolicy | None = None


def _init_worker(env_cfg: Mapping[str, Any], spec: HeuristicSpec) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_ENV, _WORKER_POLICY
    _WORKER_POLICY = HeuristicPlanningPolicy.from_spec(spec)
    _WORKER_ENV = make_env_from_cfg(cfg={"env": dict(env_cfg)}, seed=int(seed32_from(base_seed=0, stream_id=0))).env


def _worker_eval(args: tuple[int, int, int]) -> Dict[str, Any]:
    eval_steps, seed_base, deterministic = args
    if _WORKER_ENV is None or _WORKER_POLICY is None:
        raise RuntimeError("worker not initialized")
    return _planning_eval_state(
        policy=_WORKER_POLICY,
        env=_WORKER_ENV,
        eval_steps=int(eval_steps),
        seed_base=int(seed_base),
        deterministic=bool(deterministic),
        on_episode=None,
        on_step=None,
    )


def evaluate_planning_policy_parallel(
    *,
    spec: HeuristicSpec,
    env_cfg: Mapping[str, Any],
    eval_steps: int,
    seed_base: int,
    deterministic: bool,
    workers: int,
    on_episode: Optional[Callable[[int, Optional[float]], None]] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    workers = max(1, int(workers))
    eval_steps = int(eval_steps)
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be > 0, got {eval_steps}")

    if workers <= 1:
        env = make_env_from_cfg(cfg={"env": dict(env_cfg)}, seed=int(seed_base)).env
        try:
            policy = HeuristicPlanningPolicy.from_spec(spec)
            return evaluate_planning_policy(
                policy=policy,
                env=env,
                eval_steps=eval_steps,
                seed_base=int(seed_base),
                deterministic=bool(deterministic),
                on_episode=on_episode,
                on_step=on_step,
            )
        finally:
            env.close()

    per_worker = eval_steps // workers
    remainder = eval_steps % workers
    tasks: list[tuple[int, int, int]] = []
    for i in range(int(workers)):
        steps_i = int(per_worker + (1 if i < remainder else 0))
        if steps_i <= 0:
            continue
        seed_i = int(seed32_from(base_seed=int(seed_base), stream_id=int(0xE9A1 + i)))
        tasks.append((steps_i, seed_i, int(deterministic)))

    if not tasks:
        raise ValueError("no eval tasks to run")

    ctx = get_context("spawn")
    states: list[Dict[str, Any]] = []
    pool = None
    terminated = False
    prev_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(_signum, _frame) -> None:
        if pool is not None:
            pool.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        pool = ctx.Pool(
            processes=len(tasks),
            initializer=_init_worker,
            initargs=(dict(env_cfg), spec),
        )
        for state in pool.imap(_worker_eval, tasks):
            states.append(state)
            if on_step is not None:
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

    # No meaningful ordering of episodes in parallel; skip on_episode.
    _ = on_episode

    return _summarize_eval_state(
        states=states,
        eval_steps=eval_steps,
        seed_base=seed_base,
        deterministic=deterministic,
        num_envs=len(tasks),
    )


__all__ = ["evaluate_planning_policy", "evaluate_planning_policy_parallel"]
