# src/planning_rl/utils/seed.py
from __future__ import annotations


def splitmix64(x: int) -> int:
    z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFFFFFFFFFF)


def seed32_from(*, base_seed: int, stream_id: int) -> int:
    mixed = splitmix64((int(base_seed) << 32) ^ int(stream_id))
    return int(mixed & 0x7FFFFFFF)


__all__ = ["seed32_from", "splitmix64"]
