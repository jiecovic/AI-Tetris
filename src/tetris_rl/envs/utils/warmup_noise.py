# src/tetris_rl/envs/utils/warmup_noise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tetris_rl.envs.api import WarmupFn
from tetris_rl.game.core.init_rows import InitRowsApplySpec, apply_init_rows_layout


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _row_contiguous_holes(
    *,
    rng: np.random.Generator,
    w: int,
    k: int,
    holes_mean: float,
    fill_value: int,
) -> np.ndarray:
    w = int(w)
    if w <= 0:
        return np.zeros((0,), dtype=np.uint8)

    k = max(1, int(k))

    if int(fill_value) > 0:
        row = np.full((w,), int(fill_value), dtype=np.uint8)
    else:
        row = rng.integers(1, k + 1, size=(w,), dtype=np.int64).astype(np.uint8, copy=False)

    hm = max(0.0, float(holes_mean))
    nh = int(rng.poisson(lam=hm))
    nh = max(1, nh)
    nh = min(nh, w)

    if nh >= w:
        row[:] = 0
        return row

    start = int(rng.integers(0, w - nh + 1))
    row[start : start + nh] = 0
    return row


def _apply_rows(
    *,
    game: Any,
    rng: np.random.Generator,
    rows: int,
    holes_mean: float,
    fill_value: int,
) -> None:
    w = int(game.w)
    pieces = game.pieces
    k = int(len(list(pieces.kinds())))

    if rows <= 0:
        layout = np.zeros((0, w), dtype=np.uint8)
        apply_init_rows_layout(game=game, spec=InitRowsApplySpec(layout=layout))
        return

    layout = np.zeros((rows, w), dtype=np.uint8)
    for r in range(rows):
        layout[r, :] = _row_contiguous_holes(
            rng=rng,
            w=w,
            k=k,
            holes_mean=float(holes_mean),
            fill_value=int(fill_value),
        )

    apply_init_rows_layout(game=game, spec=InitRowsApplySpec(layout=layout))


# ---------------------------------------------------------------------
# Poisson rows warmup (original behavior)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class InitRowsPoissonWarmup(WarmupFn):
    enabled: bool = False
    prob: float = 0.5

    rows_mean: float = 0.0
    rows_max: int = 0

    holes_mean: float = 1.5
    fill_value: int = 0

    def __call__(self, *, game: Any, rng: np.random.Generator) -> None:
        if not self.enabled:
            return

        if _clamp01(self.prob) < 1.0 and rng.random() >= self.prob:
            return

        if self.rows_mean <= 0.0:
            rows = 0
        else:
            rows = int(rng.poisson(lam=float(self.rows_mean)))

        if self.rows_max > 0:
            rows = min(rows, int(self.rows_max))

        rows = max(0, rows)

        _apply_rows(
            game=game,
            rng=rng,
            rows=rows,
            holes_mean=self.holes_mean,
            fill_value=self.fill_value,
        )


# ---------------------------------------------------------------------
# Uniform rows warmup (new)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class InitRowsUniformWarmup(WarmupFn):
    enabled: bool = False
    prob: float = 0.5

    # rows ~ Uniform[0 .. rows_max]  (if rows_max > 0)
    rows_max: int = 0

    holes_mean: float = 1.5
    fill_value: int = 0

    def __call__(self, *, game: Any, rng: np.random.Generator) -> None:
        if not self.enabled:
            return

        if _clamp01(self.prob) < 1.0 and rng.random() >= self.prob:
            return

        if self.rows_max <= 0:
            rows = 0
        else:
            rows = int(rng.integers(0, int(self.rows_max) + 1))

        _apply_rows(
            game=game,
            rng=rng,
            rows=rows,
            holes_mean=self.holes_mean,
            fill_value=self.fill_value,
        )


__all__ = [
    "InitRowsPoissonWarmup",
    "InitRowsUniformWarmup",
]
