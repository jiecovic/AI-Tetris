# src/tetris_rl/training/imitation/collect.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from tetris_rl.datagen.schema import (
    NPZ_ACTION,
    NPZ_ACTIVE_KIND,
    NPZ_GRID,
    NPZ_NEXT_KIND,
)
from tetris_rl.datagen.shard_reader import ShardDataset
from planning_rl.utils.seed import seed32_from


@dataclass(frozen=True)
class SplitShards:
    train: List[int]
    eval: List[int]


def split_shards_modulo(
    *,
    shard_ids: Sequence[int],
    base_seed: int,
    eval_mod: int,
    eval_mod_offset: int,
    seed_offset: int,
) -> SplitShards:
    ids = [int(x) for x in shard_ids]
    ids.sort()

    m = max(1, int(eval_mod))
    off0 = int(eval_mod_offset) % m
    shift = int(seed32_from(base_seed=int(base_seed), stream_id=int(seed_offset)) % m)
    off = int((off0 + shift) % m)

    tr: List[int] = []
    ev: List[int] = []
    for sid in ids:
        if (int(sid) % m) == off:
            ev.append(int(sid))
        else:
            tr.append(int(sid))

    # --- fallback: guarantee eval shards when shards exist ---
    if ids and not ev:
        if len(ids) == 1:
            # one-shard dataset: keep train-only semantics, but allow offline eval to run
            ev = [ids[0]]
            tr = [ids[0]]
        else:
            # move 1 deterministic shard from train -> eval
            j = seed32_from(base_seed=base_seed, stream_id=0xE11A) % len(tr)
            ev.append(tr.pop(j))
            ev.sort()
            tr.sort()

    return SplitShards(train=tr, eval=ev)



def _maybe_shuffle(rng: np.random.Generator, xs: List[int], enabled: bool) -> List[int]:
    if not enabled or len(xs) <= 1:
        return xs
    out = list(xs)
    rng.shuffle(out)
    return out


def _iter_indices(
    *,
    n: int,
    rng: np.random.Generator,
    shuffle: bool,
    max_take: int,
) -> np.ndarray:
    idx = np.arange(int(n), dtype=np.int64)
    if shuffle and n > 1:
        rng.shuffle(idx)
    if max_take and idx.size > int(max_take):
        idx = idx[: int(max_take)]
    return idx


def _crop_visible_grid(grid: np.ndarray, *, crop_top_rows: int) -> np.ndarray:
    """
    Crop hidden/spawn rows from the TOP of the stored grid.

    Accepts:
      - (N,H,W) batch grids
      - (H,W) single grid

    Returns a view/slice with height reduced by crop_top_rows.
    """
    c = int(crop_top_rows)
    if c <= 0:
        return grid

    if grid.ndim == 3:
        h = int(grid.shape[1])
        if c >= h:
            raise ValueError(f"crop_top_rows={c} must be < grid height H={h}")
        return grid[:, c:, :]

    if grid.ndim == 2:
        h = int(grid.shape[0])
        if c >= h:
            raise ValueError(f"crop_top_rows={c} must be < grid height H={h}")
        return grid[c:, :]

    raise ValueError(f"grid must have ndim 2 or 3, got shape={getattr(grid, 'shape', None)!r}")


def iter_bc_batches_from_dataset(
    *,
    ds: ShardDataset,
    shard_ids: Sequence[int],
    batch_size: int,
    base_seed: int,
    shuffle_shards: bool,
    shuffle_within_shard: bool,
    max_samples: int,
    drop_last: bool,
    crop_top_rows: int = 0,
    progress_cb: Any = None,
    on_shard: Optional[Callable[[int], None]] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Yield BC batches from datagen shards.

    Output keys (BC-only):
      - grid:        (B,H,W) uint8
      - active_kind: (B,)    uint8
      - next_kind:   (B,)    uint8
      - action:      (B,)    uint8

    NOTE:
      Torch conversion (e.g. action -> torch.long for CE loss) happens downstream
      in the training pipeline, not here.

    crop_top_rows:
      If >0, removes that many rows from the TOP of each stored grid sample,
      used when dataset stored full board (visible+spawn) but policy expects visible-only.

    on_shard:
      Optional callback invoked once per shard when we start processing it.
      Useful for driving a shard-loading progress bar in the caller.
    """
    bs = max(1, int(batch_size))
    remaining: Optional[int] = int(max_samples) if int(max_samples) > 0 else None
    crop_top_rows = max(0, int(crop_top_rows))

    sids = [int(x) for x in shard_ids]
    sids.sort()

    rng_global = np.random.default_rng(int(seed32_from(base_seed=int(base_seed), stream_id=0xBCA11)))
    sids = _maybe_shuffle(rng_global, sids, bool(shuffle_shards))

    def wrap_shards(it: Iterable[int], total: int) -> Iterable[int]:
        if progress_cb is None:
            return it
        return progress_cb.wrap_shards(it, total=int(total))

    pending: Dict[str, List[np.ndarray]] = {
        NPZ_GRID: [],
        NPZ_ACTIVE_KIND: [],
        NPZ_NEXT_KIND: [],
        NPZ_ACTION: [],
    }
    pending_n = 0

    def _flush() -> Optional[Dict[str, np.ndarray]]:
        nonlocal pending_n
        if pending_n <= 0:
            return None
        b = int(pending_n)

        grid = np.concatenate(pending[NPZ_GRID], axis=0)[:b]
        ak = np.concatenate(pending[NPZ_ACTIVE_KIND], axis=0)[:b]
        nk = np.concatenate(pending[NPZ_NEXT_KIND], axis=0)[:b]
        act = np.concatenate(pending[NPZ_ACTION], axis=0)[:b]

        out: Dict[str, np.ndarray] = {
            "grid": np.asarray(grid, dtype=np.uint8),
            "active_kind": np.asarray(ak, dtype=np.uint8).reshape(-1),
            "next_kind": np.asarray(nk, dtype=np.uint8).reshape(-1),
            "action": np.asarray(act, dtype=np.uint8).reshape(-1),
        }

        for k in list(pending.keys()):
            pending[k].clear()
        pending_n = 0
        return out

    for sid in wrap_shards(sids, total=len(sids)):
        if on_shard is not None:
            try:
                on_shard(int(sid))
            except Exception:
                pass

        arrays = ds.get_shard(int(sid))

        grid = np.asarray(arrays[NPZ_GRID], dtype=np.uint8)
        grid = _crop_visible_grid(grid, crop_top_rows=crop_top_rows)

        ak = np.asarray(arrays[NPZ_ACTIVE_KIND], dtype=np.uint8).reshape(-1)
        nk = np.asarray(arrays[NPZ_NEXT_KIND], dtype=np.uint8).reshape(-1)
        act = np.asarray(arrays[NPZ_ACTION], dtype=np.uint8).reshape(-1)

        n = int(act.shape[0])
        if n <= 0:
            continue

        shard_seed = seed32_from(base_seed=int(base_seed), stream_id=int(sid))
        rng = np.random.default_rng(int(shard_seed))

        max_take_local = int(remaining) if remaining is not None else 0
        idx = _iter_indices(n=n, rng=rng, shuffle=bool(shuffle_within_shard), max_take=max_take_local)

        for j in idx:
            if remaining is not None and remaining <= 0:
                break

            jj = int(j)

            pending[NPZ_GRID].append(grid[jj : jj + 1])
            pending[NPZ_ACTIVE_KIND].append(ak[jj : jj + 1])
            pending[NPZ_NEXT_KIND].append(nk[jj : jj + 1])
            pending[NPZ_ACTION].append(act[jj : jj + 1])

            pending_n += 1
            if remaining is not None:
                remaining -= 1

            if pending_n >= bs:
                out = _flush()
                if out is not None:
                    yield out

        if remaining is not None and remaining <= 0:
            break

    if not drop_last:
        out = _flush()
        if out is not None:
            yield out


__all__ = [
    "SplitShards",
    "split_shards_modulo",
    "iter_bc_batches_from_dataset",
]
