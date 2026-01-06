# src/tetris_rl/training/imitation/spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TickUnit = Literal["samples", "updates"]


@dataclass(frozen=True)
class ImitationRunState:
    """
    Phase-local counters used for cadences (checkpoint/eval/logging).

    We keep these separate from SB3's num_timesteps because imitation is offline.
    """
    samples_seen: int = 0
    updates: int = 0


@dataclass(frozen=True)
class ImitationSplitSpec:
    """
    Deterministic shard split.

    Rule (default):
      eval iff (shard_id % eval_mod == eval_mod_offset)

    This keeps splits stable across runs and avoids row-level leakage.
    """
    eval_mod: int = 50  # ~2% eval if shards are dense
    eval_mod_offset: int = 0  # which residue is eval

    # Optional: shift based on base seed to decorrelate across experiments
    seed_offset: int = 12345


@dataclass(frozen=True)
class ImitationScheduleSpec:
    """
    Cadences for offline imitation.

    tick_unit:
      - "samples": cadences measured in samples_seen (recommended)
      - "updates": cadences measured in optimizer updates
    """
    tick_unit: TickUnit = "samples"

    # How often to save latest.zip (cadence counter depends on tick_unit).
    latest_every: int = 50_000

    # How often to run rollout eval (cadence counter depends on tick_unit).
    eval_every: int = 0

    # Optional: just for console/TB noise control
    log_every: int = 200


__all__ = [
    "TickUnit",
    "ImitationRunState",
    "ImitationSplitSpec",
    "ImitationScheduleSpec",
]
