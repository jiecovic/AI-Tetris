# src/tetris_rl/datagen/worker.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tetris_rl.config.datagen_spec import DataGenSpec
from tetris_rl.datagen.expert_factory import make_expert_from_spec
from tetris_rl.utils.seed import seed32_from

# Injected once per spawned worker process via ProcessPoolExecutor(initializer=..., initargs=...)
_WORKER_PROGRESS_QUEUE: Any = None


@dataclass(frozen=True)
class WorkerResult:
    worker_id: int
    shard_ids: list[int]
    shards_written: int
    samples_written: int


def _set_worker_progress_queue(q: Any) -> None:
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = q


def _get_progress_queue(q: Any) -> Any:
    return q if q is not None else _WORKER_PROGRESS_QUEUE


def _q_put(q: Any, ev: tuple[Any, ...]) -> None:
    if q is None:
        return
    try:
        q.put_nowait(ev)
    except Exception:
        pass


def _sample_valid_action_id_from_mask(rng: np.random.Generator, mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return -1
    j = int(rng.integers(0, idx.size))
    return int(idx[j])


def worker_generate_shards(
    *,
    worker_id: int,
    shard_ids: list[int],
    spec: DataGenSpec,
    cfg: dict[str, Any],          # resolved root cfg used by training/watch
    dataset_dir: str,
    progress_queue: Any = None,
    progress_every: int = 0,
) -> WorkerResult:
    """
    ENV-ONLY datagen worker:

      - builds the env via tetris_rl.env_bundles.factory.make_env_from_cfg (SSOT)
      - steps engine with Rust expert (action_id)
      - records env obs dict: {"grid","active_kind","next_kind"} and the action_id
      - writes NPZ shards directly (no rewardfit, no K, no extra labels)

    Shard format (npz):
      - grid:        uint8  (N, H, W)
      - active_kind: int64  (N,)
      - next_kind:   int64  (N,)
      - action:      int64  (N,)
    """
    if bool(spec.generation.labels.record_rewardfit):
        raise RuntimeError("record_rewardfit=true is disabled for the env-only datagen path (removed for now).")

    from tetris_rl.envs.factory import make_env_from_cfg

    out_dir = Path(dataset_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Expert factory SSOT (datagen-owned)
    expert_bundle = make_expert_from_spec(expert_spec=spec.expert)
    expert = expert_bundle.policy

    shards_written = 0
    samples_written = 0

    k = int(progress_every) if progress_every is not None else 0
    if k < 0:
        k = 0

    noise_enabled = bool(spec.generation.noise.enabled)
    noise_prob = float(spec.generation.noise.interleave_prob)
    noise_max_steps = int(spec.generation.noise.interleave_max_steps)

    for sid in shard_ids:
        s32 = seed32_from(base_seed=int(spec.run.seed), stream_id=int(sid))
        rng = np.random.default_rng(int(s32))

        # One env per shard (simple + robust)
        built = make_env_from_cfg(cfg=cfg, seed=int(s32))
        env = built.env

        # Determine shapes from a real reset (env is SSOT)
        ep_seed0 = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        obs, _info = env.reset(seed=ep_seed0)

        grid0 = np.asarray(obs["grid"], dtype=np.uint8)
        if grid0.ndim != 2:
            raise RuntimeError(f"env obs['grid'] must be 2D, got shape={grid0.shape}")

        H, W = int(grid0.shape[0]), int(grid0.shape[1])
        action_dim = int(getattr(env.action_space, "n", -1))
        if action_dim <= 0:
            raise RuntimeError("env.action_space must be Discrete for this datagen path")

        N = int(spec.dataset.shards.shard_steps)
        if N <= 0:
            raise ValueError(f"invalid shard_steps: {N}")

        pq = _get_progress_queue(progress_queue)
        _q_put(pq, ("start", int(worker_id), int(sid), int(N)))

        grid_buf = np.empty((N, H, W), dtype=np.uint8)
        ak_buf = np.empty((N,), dtype=np.int64)
        nk_buf = np.empty((N,), dtype=np.int64)
        act_buf = np.empty((N,), dtype=np.int64)

        filled = 0
        ep_steps = 0
        max_ep = spec.generation.episode_max_steps
        max_ep_i = None if max_ep is None else int(max_ep)

        def do_reset() -> None:
            nonlocal obs, ep_steps
            ep_steps = 0
            ep_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            obs, _ = env.reset(seed=ep_seed)

        while filled < N:
            # Optional: interleave noise steps between recorded samples
            if noise_enabled and noise_prob > 0.0 and noise_max_steps > 0:
                if float(rng.random()) < float(noise_prob):
                    nsteps = int(rng.integers(1, noise_max_steps + 1))
                    for _ in range(nsteps):
                        mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
                        aid_noise = _sample_valid_action_id_from_mask(rng, mask)
                        if aid_noise < 0:
                            do_reset()
                            break

                        # Step via env (keeps env caches consistent)
                        obs, _r, terminated, truncated, _info = env.step(int(aid_noise))
                        ep_steps += 1
                        if bool(terminated) or bool(truncated) or (max_ep_i is not None and ep_steps >= max_ep_i):
                            do_reset()
                            break

            # If ended, reset
            if bool(getattr(env, "last_state", {}).get("game_over", False)):  # defensive
                do_reset()
                continue

            # Record obs BEFORE expert action (BC convention)
            g = np.asarray(obs["grid"], dtype=np.uint8)
            if g.shape != (H, W):
                raise RuntimeError(f"obs['grid'] shape changed: got {g.shape} expected {(H, W)}")

            ak = int(obs["active_kind"])
            nk = int(obs["next_kind"])

            # Ask expert for action_id using the underlying engine (fast + SSOT).
            # Engine binding method: env.game.step_expert(policy) -> (terminated, cleared, invalid, action_id|None)
            terminated, _cleared, invalid, aid_opt = env.game.step_expert(expert)  # type: ignore[attr-defined]
            if aid_opt is None:
                do_reset()
                continue
            aid = int(aid_opt)
            if bool(invalid):
                raise RuntimeError(f"[datagen] expert produced invalid action: sid={sid} aid={aid}")

            # IMPORTANT: we stepped the engine directly => resync env caches for next obs.
            # MacroTetrisEnv already has _snapshot/_obs_from_state; use them.
            st = env._snapshot()  # type: ignore[attr-defined]
            env._last_state = st  # type: ignore[attr-defined]
            obs = env._obs_from_state(st)  # type: ignore[attr-defined]

            # Store sample
            grid_buf[filled] = g
            ak_buf[filled] = np.int64(ak)
            nk_buf[filled] = np.int64(nk)
            act_buf[filled] = np.int64(aid)

            filled += 1
            ep_steps += 1

            if k and (filled % k == 0):
                _q_put(pq, ("progress", int(worker_id), int(sid), int(filled), int(N)))

            if bool(terminated) or (max_ep_i is not None and ep_steps >= max_ep_i):
                do_reset()

        _q_put(pq, ("progress", int(worker_id), int(sid), int(N), int(N)))

        shard_path = shards_dir / f"shard_{int(sid):04d}.npz"
        if bool(spec.dataset.compression):
            np.savez_compressed(
                shard_path,
                grid=grid_buf,
                active_kind=ak_buf,
                next_kind=nk_buf,
                action=act_buf,
            )
        else:
            np.savez(
                shard_path,
                grid=grid_buf,
                active_kind=ak_buf,
                next_kind=nk_buf,
                action=act_buf,
            )

        shards_written += 1
        samples_written += int(N)

        _q_put(pq, ("done", int(worker_id), int(sid), int(N)))

        try:
            env.close()
        except Exception:
            pass

    return WorkerResult(
        worker_id=int(worker_id),
        shard_ids=[int(x) for x in shard_ids],
        shards_written=int(shards_written),
        samples_written=int(samples_written),
    )


__all__ = ["WorkerResult", "worker_generate_shards", "_set_worker_progress_queue"]
