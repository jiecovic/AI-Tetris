# src/tetris_rl/core/datagen/workers/interleave_noise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class InterleaveNoiseSpec:
    """
    Pre-label interleave noise (DAgger-lite).

    Before recording an expert-labeled sample:
      - with probability interleave_prob
      - take k random valid actions (k in [1..interleave_max_steps])
      - if episode terminates: reset (caller decides whether to also warmup after reset)
    """
    enabled: bool = False
    interleave_prob: float = 0.0
    interleave_max_steps: int = 1
    require_masks: bool = True


def _clamp_prob(p: float) -> float:
    try:
        x = float(p)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _sample_random_valid_action_id(
    *,
    rng: np.random.Generator,
    action_dim: int,
    mask: Optional[np.ndarray],
    require_masks: bool,
) -> int:
    A = int(action_dim)
    if A <= 0:
        raise RuntimeError("action_dim must be > 0 to sample actions")

    if mask is None:
        if bool(require_masks):
            raise RuntimeError("require_masks=True but action mask is unavailable")
        return int(rng.integers(low=0, high=A))

    m = np.asarray(mask, dtype=bool).reshape(-1)
    if int(m.shape[0]) != A:
        raise RuntimeError(f"mask has shape {m.shape}, expected (A,) with A={A}")

    ids = np.nonzero(m)[0]
    if ids.size <= 0:
        raise RuntimeError("action mask has zero valid actions")

    j = int(rng.integers(low=0, high=int(ids.size)))
    return int(ids[j])


def maybe_interleave_noise(
    *,
    rng: np.random.Generator,
    spec: InterleaveNoiseSpec,
    action_dim: int,
    get_mask: Callable[[], Optional[np.ndarray]],
    apply_action_no_record: Callable[[int], bool],
    do_reset: Callable[[], None],
) -> bool:
    """
    Apply interleave noise steps with probability spec.interleave_prob.

    Returns True if any interleave step was applied, else False.
    """
    if not bool(spec.enabled):
        return False

    p = _clamp_prob(spec.interleave_prob)
    if p <= 0.0:
        return False
    if float(rng.random()) >= p:
        return False

    A = int(action_dim)
    if A <= 0:
        raise RuntimeError("action_dim must be > 0 for interleave noise")

    max_k = max(1, int(spec.interleave_max_steps))
    k = int(rng.integers(low=1, high=max_k + 1))

    did_any = False
    for _ in range(k):
        mask = get_mask()
        aid = _sample_random_valid_action_id(
            rng=rng,
            action_dim=A,
            mask=mask,
            require_masks=bool(spec.require_masks),
        )
        did_any = True

        terminated = bool(apply_action_no_record(int(aid)))
        if terminated:
            do_reset()
            break

    return did_any
