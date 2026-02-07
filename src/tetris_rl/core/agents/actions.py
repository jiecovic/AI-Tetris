# src/tetris_rl/core/agents/actions.py
from __future__ import annotations

import argparse
from typing import Any

from tetris_rl.core.runtime.action_source import (
    as_action_pair,
    as_action_scalar,
    predict_action,
    sample_masked_discrete,
)


def choose_action(
    *,
    args: argparse.Namespace,
    algo_type: str,
    model: Any,
    obs: Any,
    env: Any,
    game: Any,
    expert_policy: Any,
    ga_policy: Any,
) -> Any:
    action_mode = str(getattr(env, "action_mode", "discrete")).strip().lower()

    if ga_policy is not None and (not bool(args.random_action)) and (not bool(args.heuristic_agent)):
        return ga_policy.predict(env=env)

    if bool(args.heuristic_agent):
        if expert_policy is None:
            raise RuntimeError("--heuristic-agent set but expert_policy is None")
        aid = expert_policy.action_id(game)
        if aid is None:
            aid = 0
        if action_mode == "discrete":
            return int(aid)
        rot_u, col_u = game.decode_action_id(int(aid))
        return (int(rot_u), int(col_u))

    if bool(args.random_action):
        if action_mode == "discrete":
            return int(sample_masked_discrete(env))
        return as_action_pair(env.action_space.sample())

    if model is None:
        raise RuntimeError("model is not loaded")

    pred = predict_action(algo_type=str(algo_type), model=model, obs=obs, env=env)
    if action_mode == "discrete":
        return as_action_scalar(pred)
    return as_action_pair(pred)


__all__ = ["choose_action"]
