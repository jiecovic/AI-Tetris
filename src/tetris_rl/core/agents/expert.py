# src/tetris_rl/core/agents/expert.py
from __future__ import annotations

import argparse
from typing import Any


def resolve_expert_policy_class(*, engine: Any) -> Any:
    from tetris_rl_engine import ExpertPolicy

    return ExpertPolicy


def make_expert_policy(*, args: argparse.Namespace, engine: Any) -> Any:
    ExpertPolicy = resolve_expert_policy_class(engine=engine)

    name = str(args.heuristic_policy).strip().lower()
    if name == "auto":
        name = "codemy0" if int(args.heuristic_lookahead) <= 0 else "codemy1"

    beam_w = max(1, int(args.heuristic_beam_width))
    beam_from_depth = int(args.heuristic_beam_from_depth)

    if name == "codemy0":
        return ExpertPolicy.codemy0(beam_width=int(beam_w), beam_from_depth=int(beam_from_depth))
    if name == "codemy1":
        return ExpertPolicy.codemy1(beam_width=int(beam_w), beam_from_depth=int(beam_from_depth))
    if name == "codemy2":
        return ExpertPolicy.codemy2(beam_width=int(beam_w), beam_from_depth=int(beam_from_depth))
    if name == "codemy2fast":
        return ExpertPolicy.codemy2fast(tail_weight=float(args.heuristic_tail_weight))

    raise RuntimeError(f"unknown heuristic policy: {name}")


__all__ = ["make_expert_policy", "resolve_expert_policy_class"]
