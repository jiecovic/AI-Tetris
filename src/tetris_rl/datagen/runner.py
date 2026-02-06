# src/tetris_rl/datagen/runner.py
from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pydantic import BaseModel

from tetris_rl.datagen.plan import DataGenPlan
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


def _plan_to_dict(plan: DataGenPlan) -> Dict[str, Any]:
    if isinstance(plan, BaseModel):
        return plan.model_dump(mode="json")
    return asdict(plan)


def _dataset_dir(*, plan: DataGenPlan, repo_root: Path) -> Path:
    ds = plan.dataset
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


def _read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _scan_shards_on_disk(*, dataset_dir: Path) -> dict[int, str]:
    """
    Fast scan: do NOT open shards. Just enumerate files and parse shard ids.

    Returns:
      {sid: "shards/shard_XXXX.npz"}
    """
    shards_dir = dataset_dir / "shards"
    out: dict[int, str] = {}
    if not shards_dir.is_dir():
        return out

    for p in shards_dir.glob("shard_*.npz"):
        try:
            sid = int(p.stem.split("_", 1)[1])
        except Exception:
            continue
        out[sid] = f"shards/{p.name}"
    return out


def _manifest_shard_ids(manifest: Any) -> set[int]:
    out: set[int] = set()
    shards_any = list(getattr(manifest, "shards", []) or [])
    for s in shards_any:
        if isinstance(s, dict):
            try:
                out.add(int(s.get("shard_id")))
            except Exception:
                pass
        else:
            try:
                out.add(int(getattr(s, "shard_id")))
            except Exception:
                pass
    return out


def _ensure_manifest(
    *,
    dataset_dir: Path,
    plan: DataGenPlan,
    board_h: int,
    board_w: int,
    num_kinds: int,
    action_dim: int,
    compression: bool,
    logger: Any = None,
) -> None:
    """
    Ensure manifest.json exists and is consistent with already-existing shards on disk.

    Design goals:
      - NEVER scan/load huge NPZ shards (too slow for completed datasets).
      - Use index.json / datagen_plan.json / manifest.json for cheap consistency checks.
      - Optionally verify a SINGLE shard (sid=min) to guard against wrong folder/spec.
      - Repair manifest.shards from disk listing if missing/empty/incomplete.

    Behavior:
      - If manifest.json missing -> create it (empty shards).
      - If shards exist on disk and manifest.shards is missing/incomplete -> rebuild shards list.
      - If dataset metadata (dims/compression/shard_steps/num_shards) appears inconsistent -> raise.
    """
    from tetris_rl.datagen.schema import DatasetManifest
    from tetris_rl.datagen.writer import init_manifest, read_manifest, write_manifest
    from tetris_rl.utils.seed import seed32_from

    manifest_path = dataset_dir / "manifest.json"
    index_path = dataset_dir / "index.json"
    prior_spec_path = dataset_dir / "datagen_plan.json"

    disk = _scan_shards_on_disk(dataset_dir=dataset_dir)
    disk_ids = set(disk.keys())

    # Determine expected shard_steps / num_shards from existing dataset metadata if present.
    idx = _read_json_if_exists(index_path)
    prior_spec = _read_json_if_exists(prior_spec_path)

    # Prefer dataset-local index/spec if present, else current spec.
    expected_num_shards = None
    expected_shard_steps = None
    expected_compression = None

    if isinstance(idx, dict):
        try:
            expected_num_shards = int(idx.get("num_shards"))
        except Exception:
            expected_num_shards = None
        try:
            expected_shard_steps = int(idx.get("shard_steps"))
        except Exception:
            expected_shard_steps = None
        try:
            expected_compression = bool(idx.get("compression"))
        except Exception:
            expected_compression = None

    if expected_num_shards is None:
        expected_num_shards = int(plan.dataset.shards.num_shards)
    if expected_shard_steps is None:
        expected_shard_steps = int(plan.dataset.shards.shard_steps)
    if expected_compression is None:
        expected_compression = bool(compression)

    # If manifest is missing, create it (but don't nuke anything).
    if not manifest_path.exists():
        m0 = init_manifest(
            name=str(plan.dataset.name),
            board_h=int(board_h),
            board_w=int(board_w),
            num_kinds=int(num_kinds),
            action_dim=int(action_dim),
            compression=bool(compression),
        )
        write_manifest(dataset_dir=dataset_dir, manifest=m0, overwrite=False)
        if logger:
            logger.info("[datagen] wrote new manifest.json")
        # Continue: if shards already exist, we likely want to fill shards list.

    # Load manifest (or rebuild minimal on parse error).
    try:
        m = read_manifest(dataset_dir=dataset_dir)
    except Exception:
        m = init_manifest(
            name=str(plan.dataset.name),
            board_h=int(board_h),
            board_w=int(board_w),
            num_kinds=int(num_kinds),
            action_dim=int(action_dim),
            compression=bool(compression),
        )

    # Cheap consistency checks against existing metadata / current env-derived dims.
    # If this is a completed folder, we want to fail loudly if the user points at the wrong run/spec.
    if disk_ids:
        # If index.json says a different num_shards than disk has, that's suspicious but not fatal
        # (could be partially complete). Still useful:
        if expected_num_shards is not None and expected_num_shards > 0:
            # Only error if disk has MORE than expected (wrong folder)
            if len(disk_ids) > int(expected_num_shards):
                raise RuntimeError(
                    f"[datagen] dataset appears inconsistent: disk_shards={len(disk_ids)} "
                    f"> expected_num_shards={expected_num_shards} (index/spec mismatch?)"
                )

        # Compare compression in manifest/index vs current.
        try:
            m_comp = bool(getattr(m, "compression"))
        except Exception:
            m_comp = bool(compression)
        if bool(m_comp) != bool(expected_compression):
            raise RuntimeError(
                f"[datagen] compression mismatch for existing dataset: "
                f"manifest={m_comp} expected={expected_compression} (refusing to repair)"
            )

        # Compare dims in manifest vs current env-derived dims.
        # If the manifest was clobbered, these may still be correct; if not, we should fail.
        try:
            if int(getattr(m, "board_h")) != int(board_h) or int(getattr(m, "board_w")) != int(board_w):
                raise RuntimeError(
                    f"[datagen] board shape mismatch for existing dataset: "
                    f"manifest=({getattr(m,'board_h',None)},{getattr(m,'board_w',None)}) "
                    f"expected=({board_h},{board_w}) (refusing to repair)"
                )
            if int(getattr(m, "num_kinds")) != int(num_kinds):
                raise RuntimeError(
                    f"[datagen] num_kinds mismatch for existing dataset: "
                    f"manifest={getattr(m,'num_kinds',None)} expected={num_kinds} (refusing to repair)"
                )
            if int(getattr(m, "action_dim")) != int(action_dim):
                raise RuntimeError(
                    f"[datagen] action_dim mismatch for existing dataset: "
                    f"manifest={getattr(m,'action_dim',None)} expected={action_dim} (refusing to repair)"
                )
        except AttributeError:
            # If manifest is extremely broken, we'll rebuild below.
            pass

        # Optional single-shard verification (cheap-ish): validate one shard matches expected_shard_steps and dims.
        # This is the safety valve that prevents us from blindly trusting shard_steps without inspecting all files.
        try:
            sid0 = min(disk_ids)
            p0 = dataset_dir / disk[sid0]
            with np.load(str(p0), allow_pickle=False) as z:
                g = z["grid"]
                if int(g.shape[0]) != int(expected_shard_steps):
                    raise RuntimeError(
                        f"[datagen] shard_steps mismatch: shard[{sid0}] has N={int(g.shape[0])} "
                        f"expected={expected_shard_steps} (refusing to repair)"
                    )
                if int(g.shape[1]) != int(board_h) or int(g.shape[2]) != int(board_w):
                    raise RuntimeError(
                        f"[datagen] shard grid shape mismatch: shard[{sid0}] grid.shape={tuple(g.shape)} "
                        f"expected=({expected_shard_steps},{board_h},{board_w}) (refusing to repair)"
                    )
        except KeyError:
            pass
        except Exception as e:
            # If the single shard check fails, it's safer to hard-error.
            raise

    manifest_ids = _manifest_shard_ids(m)

    # If no shards on disk, nothing to repair.
    if not disk_ids:
        return

    # If manifest already covers all disk shards and has entries, keep it.
    if disk_ids.issubset(manifest_ids) and list(getattr(m, "shards", []) or []):
        return

    # Rebuild shard list using expected_shard_steps (NOT by opening all NPZs).
    rebuilt: list[dict[str, Any]] = []
    for sid in sorted(disk_ids):
        seed = int(seed32_from(base_seed=int(plan.run.seed), stream_id=int(sid)))
        rebuilt.append(
            {
                "shard_id": int(sid),
                "file": str(disk[sid]),
                "num_samples": int(expected_shard_steps),
                "seed": seed,
            }
        )

    if is_dataclass(m):
        m_dict = asdict(m)
    else:
        m_dict = dict(m)

    m_dict["name"] = str(plan.dataset.name)
    m_dict["board_h"] = int(board_h)
    m_dict["board_w"] = int(board_w)
    m_dict["num_kinds"] = int(num_kinds)
    m_dict["action_dim"] = int(action_dim)
    m_dict["compression"] = bool(compression)
    m_dict["shards"] = rebuilt

    updated = DatasetManifest(**m_dict)
    write_manifest(dataset_dir=dataset_dir, manifest=updated, overwrite=True)

    if logger:
        logger.warning(
            "[datagen] repaired manifest.json from disk shards (fast): disk=%d manifest(before)=%d shard_steps=%d",
            len(disk_ids),
            len(manifest_ids),
            int(expected_shard_steps),
        )


def run_datagen(
    *,
    plan: DataGenPlan,
    cfg: dict[str, Any],
    repo_root: Path,
    logger: Any = None,
) -> Path:
    """
    BC-only datagen runner.

    Writes:
      - datagen_plan.json
      - datagen_cfg.json
      - manifest.json        (BC-minimal schema contract)
      - index.json
      - shards/shard_XXXX.npz
    """
    from tetris_rl.datagen.worker import worker_generate_shards
    from tetris_rl.envs.factory import make_env_from_cfg
    from tetris_rl.utils.seed import seed32_from

    if not isinstance(cfg, dict):
        raise TypeError(f"run_datagen(cfg=...) must be dict, got {type(cfg)!r}")

    dataset_dir = _dataset_dir(plan=plan, repo_root=repo_root)
    _ensure_dataset_dir(dataset_dir=dataset_dir)

    ds = plan.dataset
    run = plan.run

    num_shards = int(ds.shards.num_shards)
    shard_steps = int(ds.shards.shard_steps)
    num_workers = max(1, int(run.num_workers))
    compression = bool(ds.compression)

    if num_shards <= 0 or shard_steps <= 0:
        raise ValueError("num_shards and shard_steps must be > 0")

    # ------------------------------------------------------------------
    # persist inputs (debug / reproducibility only)
    # ------------------------------------------------------------------
    write_json(dataset_dir / "datagen_plan.json", _plan_to_dict(plan))
    write_json(dataset_dir / "datagen_cfg.json", cfg)

    # ------------------------------------------------------------------
    # manifest.json (create if missing; repair if shards exist but manifest empty)
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

    _ensure_manifest(
        dataset_dir=dataset_dir,
        plan=plan,
        board_h=board_h,
        board_w=board_w,
        num_kinds=num_kinds,
        action_dim=action_dim,
        compression=compression,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # resume bookkeeping (disk is authoritative for resume)
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
                        plan=plan,
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
                            plan=plan,
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
