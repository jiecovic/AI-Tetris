# src/tetris_rl/datagen/runner.py
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from tetris_rl.config.datagen_spec import DataGenSpec
from tetris_rl.datagen.progress import MultiWorkerProgress
from tetris_rl.utils.file_io import write_json


def _best_effort_close_queue(q: Any) -> None:
    try:
        q.cancel_join_thread()
    except Exception:
        pass
    try:
        q.close()
    except Exception:
        pass


def _spec_to_dict(spec: DataGenSpec) -> Dict[str, Any]:
    return asdict(spec)


def _dataset_dir(*, spec: DataGenSpec, repo_root: Path) -> Path:
    ds = spec.dataset
    return (Path(repo_root) / ds.out_root / ds.name).resolve()


def _existing_shard_ids(shards_dir: Path) -> set[int]:
    if not shards_dir.is_dir():
        return set()
    out: set[int] = set()
    for p in shards_dir.glob("shard_*.npz"):
        try:
            out.add(int(p.stem.split("_", 1)[1]))
        except Exception:
            pass
    return out


def _ensure_dataset_dir(*, dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "shards").mkdir(parents=True, exist_ok=True)


def _hard_kill_executor(ex: ProcessPoolExecutor, futures: list[Any] | None = None) -> None:
    if futures is not None:
        for fut in futures:
            try:
                fut.cancel()
            except Exception:
                pass

    procs = getattr(ex, "_processes", None)
    if isinstance(procs, dict):
        for p in list(procs.values()):
            try:
                p.terminate()
            except Exception:
                pass
        for p in list(procs.values()):
            try:
                p.join(timeout=0.5)
            except Exception:
                pass

    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)
    except Exception:
        pass


def _init_worker_progress_queue(q: Any) -> None:
    # Windows: Ctrl+C is broadcast to all console processes.
    # Only parent handles SIGINT.
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    from tetris_rl.datagen.worker import _set_worker_progress_queue

    _set_worker_progress_queue(q)


def run_datagen(
    *,
    spec: DataGenSpec,
    cfg: dict[str, Any],
    repo_root: Path,
    logger: Any = None,
) -> Path:
    """
    BC-only datagen runner.

    Writes:
      - datagen_spec.json
      - datagen_cfg.json
      - manifest.json        (BC-minimal schema contract)
      - index.json
      - shards/shard_XXXX.npz
    """
    from tetris_rl.datagen.worker import worker_generate_shards
    from tetris_rl.datagen.writer import init_manifest, write_manifest
    from tetris_rl.envs.factory import make_env_from_cfg
    from tetris_rl.utils.seed import seed32_from

    if not isinstance(cfg, dict):
        raise TypeError(f"run_datagen(cfg=...) must be dict, got {type(cfg)!r}")

    dataset_dir = _dataset_dir(spec=spec, repo_root=repo_root)
    _ensure_dataset_dir(dataset_dir=dataset_dir)

    ds = spec.dataset
    run = spec.run

    num_shards = int(ds.shards.num_shards)
    shard_steps = int(ds.shards.shard_steps)
    num_workers = max(1, int(run.num_workers))
    compression = bool(ds.compression)

    if num_shards <= 0 or shard_steps <= 0:
        raise ValueError("num_shards and shard_steps must be > 0")

    # ------------------------------------------------------------------
    # persist inputs (debug / reproducibility only)
    # ------------------------------------------------------------------
    write_json(dataset_dir / "datagen_spec.json", _spec_to_dict(spec))
    write_json(dataset_dir / "datagen_cfg.json", cfg)

    # ------------------------------------------------------------------
    # manifest.json (BC-minimal, written ONCE before workers start)
    # ------------------------------------------------------------------
    seed0 = int(seed32_from(base_seed=int(run.seed), stream_id=0xDA7A))
    built = make_env_from_cfg(cfg=cfg, seed=seed0)
    env = built.env
    try:
        obs_space = env.observation_space
        act_space = env.action_space

        board_h = int(obs_space["grid"].shape[0])
        board_w = int(obs_space["grid"].shape[1])
        num_kinds = int(obs_space["active_kind"].n)
        action_dim = int(act_space.n)
    finally:
        try:
            env.close()
        except Exception:
            pass

    manifest = init_manifest(
        name=str(ds.name),
        board_h=board_h,
        board_w=board_w,
        num_kinds=num_kinds,
        action_dim=action_dim,
        compression=compression,
    )
    write_manifest(dataset_dir=dataset_dir, manifest=manifest)

    # ------------------------------------------------------------------
    # resume bookkeeping
    # ------------------------------------------------------------------
    shards_dir = dataset_dir / "shards"
    existing = _existing_shard_ids(shards_dir)
    expected = set(range(num_shards))
    missing = sorted(expected - existing)

    mode = "resume" if existing else "new"
    progress_every = 1 if num_workers <= 1 else max(
        1, int(getattr(run, "progress_update_every_k", 2000))
    )

    # ------------------------------------------------------------------
    # generate shards
    # ------------------------------------------------------------------
    if missing:
        with MultiWorkerProgress(
            total_shards=num_shards,
            shard_steps=shard_steps,
            num_slots=num_workers,
            already_done=len(existing),
        ) as prog:
            if num_workers == 1:
                for sid in missing:
                    worker_generate_shards(
                        worker_id=0,
                        shard_ids=[sid],
                        spec=spec,
                        cfg=cfg,
                        dataset_dir=str(dataset_dir),
                        progress_queue=prog.queue,
                        progress_every=progress_every,
                    )
            else:
                ex: ProcessPoolExecutor | None = None
                futures: list[Any] = []

                ctx = mp.get_context("spawn")
                try:
                    ex = ProcessPoolExecutor(
                        max_workers=num_workers,
                        mp_context=ctx,
                        initializer=_init_worker_progress_queue,
                        initargs=(prog.queue,),
                    )

                    for i, sid in enumerate(missing):
                        fut = ex.submit(
                            worker_generate_shards,
                            worker_id=i % num_workers,
                            shard_ids=[sid],
                            spec=spec,
                            cfg=cfg,
                            dataset_dir=str(dataset_dir),
                            progress_queue=None,
                            progress_every=progress_every,
                        )
                        futures.append(fut)

                    pending = set(futures)
                    while pending:
                        done, pending = wait(
                            pending, timeout=0.25, return_when=FIRST_COMPLETED
                        )
                        for fut in done:
                            fut.result()

                    ex.shutdown(wait=True)
                    ex = None

                except KeyboardInterrupt:
                    if logger:
                        logger.warning("[datagen] interrupted; stopping workers")
                    if ex is not None:
                        _hard_kill_executor(ex, futures)
                    _best_effort_close_queue(prog.queue)
                    raise
                except Exception:
                    if logger:
                        logger.exception("[datagen] worker failure")
                    if ex is not None:
                        _hard_kill_executor(ex, futures)
                    _best_effort_close_queue(prog.queue)
                    raise
                finally:
                    if ex is not None:
                        _hard_kill_executor(ex, futures)

    # ------------------------------------------------------------------
    # index.json (summary only, not schema)
    # ------------------------------------------------------------------
    write_json(
        dataset_dir / "index.json",
        {
            "dataset_dir": str(dataset_dir),
            "mode": mode,
            "num_shards": int(num_shards),
            "shard_steps": int(shard_steps),
            "samples_total": int(num_shards * shard_steps),
            "workers": int(num_workers),
            "compression": bool(compression),
        },
    )

    if logger:
        logger.info("[datagen] finished: %s", dataset_dir)

    return dataset_dir


__all__ = ["run_datagen"]
