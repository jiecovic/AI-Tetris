# src/tetris_rl/utils/seed.py
from __future__ import annotations

import numpy as np

"""
Deterministic seed utilities shared across the codebase.

Purpose:
  - Provide a single, stable implementation of splitmix64
  - Derive reproducible 32-bit seeds from a base seed + stream id
    (e.g. shard_id, worker_id, env_id, eval offset, etc.)

This module MUST remain side-effect free and deterministic.
No RNG state is stored here.
"""


def splitmix64(x: int) -> int:
    """
    Stateless 64-bit SplitMix hash.

    Properties:
      - Deterministic
      - Good bit diffusion
      - Cheap
      - Suitable for deriving independent RNG streams

    Input:
      x : int (treated as unsigned 64-bit)

    Output:
      uint64 encoded as Python int
    """
    z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFFFFFFFFFF)

def splitmix64_np(x: np.ndarray) -> np.ndarray:
    """
    Vectorized SplitMix64 for uint64 arrays.

    Input:
      x: array-like (will be cast to uint64)

    Output:
      uint64 ndarray, same shape
    """
    z = np.asarray(x, dtype=np.uint64)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = z ^ (z >> np.uint64(31))
    return z & np.uint64(0xFFFFFFFFFFFFFFFF)


def seed32_from(*, base_seed: int, stream_id: int) -> int:
    """
    Derive a deterministic 32-bit seed from a base seed and a stream id.

    Typical usage:
      - shard seed:  seed32_from(base_seed=run_seed, stream_id=shard_id)
      - worker seed: seed32_from(base_seed=run_seed, stream_id=worker_id)
      - eval seed:   seed32_from(base_seed=run_seed, stream_id=seed_offset)

    Guarantees:
      - Same (base_seed, stream_id) -> same seed
      - Different stream_id -> decorrelated streams
      - Result fits in signed 32-bit int (safe for NumPy, Gym, SB3)

    Returns:
      int in [0, 2^31 - 1]
    """
    mixed = splitmix64((int(base_seed) << 32) ^ int(stream_id))
    return int(mixed & 0x7FFFFFFF)


__all__ = [
    "splitmix64",
    "seed32_from",
]
