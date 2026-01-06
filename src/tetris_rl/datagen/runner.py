# src/tetris_rl/datagen/runner.py
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

from tetris_rl.config.datagen_spec import DataGenSpec
from tetris_rl.datagen.progress import MultiWorkerProgress
from tetris_rl.datagen.shardinfo import build_expected_shard_infos
from tetris_rl.datagen.writer import append_shard_to_manifest, init_manifest, write_manifest
from tetris_rl.utils.file_io import write_json


def _spec_to_dict(spec: DataGenSpec) -> Dict[str, Any]:
    # explicit, stable snapshot
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


def _hard_kill_executor(
        ex: ProcessPoolExecutor,
        futures: list[Any] | None = None,
) -> None:
    # 1) Cancel futures (Python-side)
    if futures is not None:
        for fut in futures:
            try:
                fut.cancel()
            except Exception:
                pass

    # 2) Terminate worker processes (OS-side)
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

    # 3) Shutdown executor last
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)
    except Exception:
        pass


def _init_worker_progress_queue(q: Any) -> None:
    """
    Called once inside each spawned worker process.

    We inject the progress queue via ProcessPoolExecutor(initializer=..., initargs=(q,))
    because on Windows 'spawn' you cannot pickle/pass a multiprocessing.Queue per-task.
    """
    from tetris_rl.datagen.worker import _set_worker_progress_queue

    _set_worker_progress_queue(q)


def _game_metadata_from_spec(*, spec: DataGenSpec) -> Tuple[int, int, int, int]:
    """
    Returns (board_h, board_w, num_kinds, max_rots).

    Policy:
      - board geometry + kinds are derived from the concrete game assets.
      - max_rots is derived from the piece asset rotations (PieceSet is source of truth).
    """
    from tetris_rl.game.factory import make_game_from_spec

    game = make_game_from_spec(spec.game)

    board_h = int(game.h)
    board_w = int(game.w)
    if board_h <= 0 or board_w <= 0:
        raise RuntimeError(f"invalid board dims from game: h={board_h}, w={board_w}")

    pieces = game.pieces
    kinds = list(pieces.kinds()) if hasattr(pieces, "kinds") else []
    num_kinds = max(1, len(kinds))

    try:
        max_rots = int(pieces.max_rotations())
    except Exception as e:
        raise RuntimeError("pieces.max_rotations() failed; PieceSet must define this") from e

    if max_rots <= 0:
        raise RuntimeError(f"invalid max_rots derived from pieces: {max_rots}")

    return board_h, board_w, int(num_kinds), int(max_rots)


def _shard_info_for_sid(
        *,
        sid: int,
        num_shards: int,
        shard_steps: int,
        base_seed: int,
        episode_max_steps: Any,
):
    infos = build_expected_shard_infos(
        num_shards=num_shards,
        shard_steps=shard_steps,
        base_seed=base_seed,
        episode_max_steps=episode_max_steps,
    )
    for si in infos:
        if int(getattr(si, "shard_id", -1)) == int(sid):
            return si
    raise RuntimeError(f"build_expected_shard_infos did not produce shard_id={sid}")


def run_datagen(*, spec: DataGenSpec, repo_root: Path, logger: Any = None) -> Path:
    from tetris_rl.datagen.worker import worker_generate_shards

    dataset_dir = _dataset_dir(spec=spec, repo_root=repo_root)
    _ensure_dataset_dir(dataset_dir=dataset_dir)

    ds = spec.dataset
    run = spec.run
    gen = spec.generation

    num_shards = int(ds.shards.num_shards)
    shard_steps = int(ds.shards.shard_steps)
    num_workers = max(1, int(run.num_workers))

    if num_shards <= 0 or shard_steps <= 0:
        raise ValueError("num_shards and shard_steps must be > 0")

    compression = bool(ds.compression)

    spec_dict = _spec_to_dict(spec)
    write_json(dataset_dir / "datagen_spec.json", spec_dict)

    # ------------------------------------------------------------------
    # Create the dataset envelope immediately so partial datasets load.
    # ------------------------------------------------------------------
    board_h, board_w, num_kinds, max_rots = _game_metadata_from_spec(spec=spec)
    action_dim = max_rots * board_w

    manifest = init_manifest(
        name=ds.name,
        board_h=board_h,
        board_w=board_w,
        num_kinds=num_kinds,
        action_dim=action_dim,
        max_rots=max_rots,
        pieces=spec.game.pieces,
        piece_rule=spec.game.piece_rule,
        compression=compression,
        datagen_spec=spec_dict,
    )

    # Ensure shards list exists and starts with whatever is already on disk (resume)
    shards_dir = dataset_dir / "shards"
    existing = _existing_shard_ids(shards_dir)

    for sid in sorted(existing):
        si = _shard_info_for_sid(
            sid=sid,
            num_shards=num_shards,
            shard_steps=shard_steps,
            base_seed=run.seed,
            episode_max_steps=gen.episode_max_steps,
        )
        try:
            manifest.shards.append(si)
        except Exception:
            raise RuntimeError("manifest.shards is not appendable; DatasetManifest must define shards as a list")

    write_manifest(dataset_dir=dataset_dir, manifest=manifest)

    expected = set(range(num_shards))
    missing = sorted(expected - existing)

    mode = "resume" if existing else "new"
    if num_workers <= 1:
        progress_every = 1
    else:
        cfg_k = int(getattr(run, "progress_update_every_k", 1))
        progress_every = max(1, cfg_k)

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
                        dataset_dir=str(dataset_dir),
                        progress_queue=prog.queue,  # OK in-process
                        progress_every=progress_every,
                    )

                    si = _shard_info_for_sid(
                        sid=sid,
                        num_shards=num_shards,
                        shard_steps=shard_steps,
                        base_seed=run.seed,
                        episode_max_steps=gen.episode_max_steps,
                    )
                    append_shard_to_manifest(dataset_dir=dataset_dir, shard=si)

            else:
                ex: ProcessPoolExecutor | None = None
                futures: list[Any] = []
                fut_to_sid: dict[Any, int] = {}

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
                            dataset_dir=str(dataset_dir),
                            progress_queue=None,  # injected via initializer
                            progress_every=progress_every,
                        )
                        futures.append(fut)
                        fut_to_sid[fut] = int(sid)

                    pending = set(futures)
                    while pending:
                        done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                        for fut in done:
                            fut.result()  # surface worker exceptions

                            sid = int(fut_to_sid.get(fut, -1))
                            if sid >= 0:
                                si = _shard_info_for_sid(
                                    sid=sid,
                                    num_shards=num_shards,
                                    shard_steps=shard_steps,
                                    base_seed=run.seed,
                                    episode_max_steps=gen.episode_max_steps,
                                )
                                append_shard_to_manifest(dataset_dir=dataset_dir, shard=si)

                    ex.shutdown(wait=True, cancel_futures=False)
                    ex = None

                except KeyboardInterrupt:
                    if logger:
                        logger.warning("[datagen] interrupted; stopping workers...")
                    if ex is not None:
                        _hard_kill_executor(ex, futures)
                    raise
                except Exception:
                    if logger:
                        logger.exception("[datagen] worker failure; stopping remaining workers...")
                    if ex is not None:
                        _hard_kill_executor(ex, futures)
                    raise
                finally:
                    if ex is not None:
                        _hard_kill_executor(ex, futures)

    # Final integrity check for fully completed datasets (keeps behavior identical at end)
    for sid in range(num_shards):
        p = shards_dir / f"shard_{sid:04d}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"missing shard after generation: {p}")

    write_json(
        dataset_dir / "index.json",
        {
            "dataset_dir": str(dataset_dir),
            "mode": mode,
            "num_shards": num_shards,
            "shard_steps": shard_steps,
            "samples_total": num_shards * shard_steps,
            "workers": num_workers,
            "compression": compression,
        },
    )

    if logger:
        logger.info("[datagen] finished: %s", dataset_dir)

    return dataset_dir
