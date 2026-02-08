# src/tetris_rl/core/utils/seed.py
from __future__ import annotations

import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional at runtime
    torch = None


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and Torch once per run for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["seed_all"]
