# src/tetris_rl/core/training/evaluation/eval_planning.py
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from tetris_rl.core.training.metrics import StatsAccumulator, StatsAccumulatorConfig


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


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
    ep_max_levels: list[Optional[float]] = []

    next_seed = int(seed_base)
    obs, info = env.reset(seed=int(next_seed))
    _ = obs
    _ = info
    next_seed += 1

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

        if not (terminated or truncated):
            continue

        completed_episodes += 1
        ep_returns.append(float(cur_ep_reward))
        ep_steps.append(int(cur_ep_steps))

        game = info.get("game") if isinstance(info, Mapping) else None
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

        cur_ep_reward = 0.0
        cur_ep_steps = 0

        _obs2, _info2 = env.reset(seed=int(next_seed))
        _ = _obs2
        _ = _info2
        next_seed += 1

    out: Dict[str, Any] = {}
    out.update(acc.summarize())

    out["eval/steps"] = int(acc.steps)
    out["eval/deterministic"] = bool(deterministic)
    out["eval/seed_base"] = int(seed_base)
    out["eval/num_envs"] = 1
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

    return out


__all__ = ["evaluate_planning_policy"]
