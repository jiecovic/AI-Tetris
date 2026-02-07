# src/tetris_rl/datagen/shardinfo.py
from __future__ import annotations

from typing import Optional, List

from tetris_rl.datagen.schema import ShardInfo
from planning_rl.utils.seed import seed32_from


def build_expected_shard_infos(
        *,
        num_shards: int,
        shard_steps: int,
        base_seed: int,
        episode_max_steps: Optional[int],
) -> List[ShardInfo]:
    """
    Deterministically build per-shard metadata for manifest.json.

    This is what makes "resume" easy:
      - shard seed depends ONLY on (base_seed, shard_id)
      - therefore shard contents are independent of worker count / scheduling
      - you can safely generate shards in any order, any parallelism, and still get
        exactly the same dataset as a single sequential run.
    """
    infos: List[ShardInfo] = []
    for sid in range(int(num_shards)):
        s32 = seed32_from(base_seed=int(base_seed), stream_id=int(sid))
        infos.append(
            ShardInfo(
                shard_id=int(sid),
                file=f"shards/shard_{int(sid):04d}.npz",
                num_samples=int(shard_steps),
                seed=int(s32),
                episode_max_steps=episode_max_steps,
            )
        )
    infos.sort(key=lambda s: int(s.shard_id))
    return infos
