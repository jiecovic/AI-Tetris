# src/tetris_rl/training/imitation/runner.py
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm.rich import tqdm  # <-- rich tqdm (NOT tqdm.auto)

from tetris_rl.runs.config import RunConfig
from tetris_rl.training.config import TrainConfig
from tetris_rl.datagen.shard_reader import ShardDataset
from tetris_rl.runs.checkpoint_manager import CheckpointManager, CheckpointPaths
from tetris_rl.training.evaluation.eval_checkpoint_core import EvalCheckpointCore, EvalCheckpointCoreSpec
from tetris_rl.training.evaluation.latest_checkpoint_core import LatestCheckpointCore, LatestCheckpointCoreSpec
from tetris_rl.training.imitation.bc_train import BCTrainSpec, bc_eval_stream, bc_train_stream
from tetris_rl.training.imitation.collect import iter_bc_batches_from_dataset, split_shards_modulo
from tetris_rl.training.imitation.spec import ImitationRunState, ImitationScheduleSpec, ImitationSplitSpec
from tetris_rl.utils.paths import repo_root
from tetris_rl.utils.seed import seed32_from


@dataclass(frozen=True)
class ImitationRunnerSpec:
    dataset_dir: Path
    epochs: int
    batch_size: int
    max_samples: int

    shuffle_shards: bool = True
    shuffle_within_shard: bool = True
    drop_last: bool = False

    split: ImitationSplitSpec = ImitationSplitSpec()
    schedule: ImitationScheduleSpec = ImitationScheduleSpec()


def _resolve_dataset_dir(repo: Path, dataset_dir: str) -> Path:
    p = Path(str(dataset_dir))
    return p if p.is_absolute() else (Path(repo) / p).resolve()


def _tick_counter(state: ImitationRunState, tick_unit: str) -> int:
    u = str(tick_unit).strip().lower()
    if u == "updates":
        return int(state.updates)
    return int(state.samples_seen)


def _infer_policy_grid_hw(model: Any) -> Tuple[int, int]:
    policy = getattr(model, "policy", None)
    obs_space = getattr(policy, "observation_space", None)
    if obs_space is None:
        obs_space = getattr(model, "observation_space", None)

    spaces = getattr(obs_space, "spaces", None)
    if not isinstance(spaces, dict) or "grid" not in spaces:
        raise ValueError(
            "[imitation] policy obs_space must be a Dict space with key 'grid'. "
            f"Got obs_space={type(obs_space).__name__} with keys={list(spaces.keys()) if isinstance(spaces, dict) else None}"
        )

    grid_space = spaces["grid"]
    shape = getattr(grid_space, "shape", None)
    if not shape or len(shape) != 2:
        raise ValueError(
            "[imitation] policy obs_space['grid'] must have shape (H,W). "
            f"Got shape={shape!r} (space={type(grid_space).__name__})"
        )
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"[imitation] invalid policy grid shape: {(h, w)!r}")
    return h, w


def _compute_dataset_grid_crop(*, ds_h: int, ds_w: int, obs_h: int, obs_w: int, dataset_dir: Path) -> int:
    if int(ds_w) != int(obs_w):
        raise ValueError(
            "[imitation] dataset width does not match policy.\n"
            f"  dataset_dir: {str(dataset_dir)}\n"
            f"  manifest board_w: {int(ds_w)}\n"
            f"  policy grid W:    {int(obs_w)}\n"
            "Fix: use a dataset generated with the same board width as the current env/model config."
        )

    if int(obs_h) == int(ds_h):
        return 0

    if int(obs_h) < int(ds_h):
        return int(ds_h) - int(obs_h)

    raise ValueError(
        "[imitation] policy expects a taller grid than what the dataset provides.\n"
        f"  dataset_dir: {str(dataset_dir)}\n"
        f"  manifest board_h: {int(ds_h)}\n"
        f"  policy grid H:    {int(obs_h)}\n"
        "Fix: either (a) train/eval with an env that uses the dataset's grid height, or (b) regenerate the dataset."
    )


def _estimate_epoch_batches(*, ds: ShardDataset, train_sids: list[int], batch_size: int, max_samples: int) -> int:
    bs = max(1, int(batch_size))
    lim = int(max_samples) if int(max_samples) > 0 else 0

    counts: dict[int, int] = {int(r.shard_id): int(r.num_samples) for r in ds.shards}
    total = 0
    for sid in train_sids:
        total += int(counts.get(int(sid), 0))

    if lim > 0:
        total = min(total, lim)

    if total <= 0:
        return 0
    return int(ceil(float(total) / float(bs)))


def run_imitation(
    *,
    cfg: Dict[str, Any],
    model: Any,
    train_cfg: TrainConfig,
    run_cfg: RunConfig,
    run_dir: Path,
    repo: Optional[Path] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    repo_p = Path(repo) if repo is not None else repo_root()

    im = train_cfg.imitation
    if not bool(im.enabled):
        if logger is not None:
            try:
                logger.info("[imitation] disabled")
            except Exception:
                pass
        return {"enabled": False}

    ds_dir = _resolve_dataset_dir(repo_p, str(im.dataset_dir))
    ds = ShardDataset(dataset_dir=ds_dir)

    ds_h = int(ds.manifest.board_h)
    ds_w = int(ds.manifest.board_w)
    obs_h, obs_w = _infer_policy_grid_hw(model)
    crop_top_rows = _compute_dataset_grid_crop(ds_h=ds_h, ds_w=ds_w, obs_h=obs_h, obs_w=obs_w, dataset_dir=ds_dir)

    if logger is not None:
        try:
            if crop_top_rows > 0:
                logger.info(
                    "[imitation] grid adapter: dataset grid=(%d,%d) -> policy grid=(%d,%d) by cropping top_rows=%d",
                    ds_h,
                    ds_w,
                    obs_h,
                    obs_w,
                    crop_top_rows,
                )
            else:
                logger.info("[imitation] grid adapter: dataset grid matches policy grid (%d,%d)", obs_h, obs_w)
        except Exception:
            pass

    shard_ids = [int(s.shard_id) for s in ds.shards]
    split = split_shards_modulo(
        shard_ids=shard_ids,
        base_seed=int(run_cfg.seed),
        eval_mod=200,
        eval_mod_offset=0,
        seed_offset=12345,
    )
    train_sids = split.train
    eval_sids = split.eval

    manager = CheckpointManager(paths=CheckpointPaths(checkpoint_dir=Path(run_dir) / "checkpoints"), verbose=0)
    manager.ensure_dir()
    manager.load_state()

    sched = ImitationScheduleSpec(
        tick_unit="samples",
        latest_every=int(train_cfg.checkpoints.latest_every),
        eval_every=int(train_cfg.eval.eval_every),
        log_every=50,
    )

    latest_core = LatestCheckpointCore(
        spec=LatestCheckpointCoreSpec(
            checkpoint_dir=Path(manager.paths.checkpoint_dir),
            latest_every=int(sched.latest_every),
            verbose=0,
        ),
        emit=(lambda s: logger.info(s)) if logger is not None else None,
    )

    eval_core = EvalCheckpointCore(
        spec=EvalCheckpointCoreSpec(
            checkpoint_dir=Path(manager.paths.checkpoint_dir),
            eval_every=int(sched.eval_every),
            run_cfg=run_cfg,
            eval=train_cfg.eval,
            base_seed=int(run_cfg.seed),
            table_header_every=10,
            progress_unit=str(sched.tick_unit),
            verbose=1,
        ),
        cfg=cfg,
        emit=(lambda s: logger.info(s)) if logger is not None else None,
        log_scalar=None,
    )

    latest_core.init(progress_step=0)
    eval_core.init(progress_step=0)

    state = ImitationRunState(samples_seen=0, updates=0)
    device = str(run_cfg.device).strip() if str(run_cfg.device).strip() else "cpu"

    if logger is not None:
        try:
            logger.info(
                "[imitation] dataset=%s  shards(train=%d eval=%d)  batch=%d  epochs=%d  max_samples=%d",
                str(ds_dir),
                int(len(train_sids)),
                int(len(eval_sids)),
                int(im.batch_size),
                int(im.epochs),
                int(im.max_samples),
            )
            logger.info(
                "[imitation] cadences: latest_every=%d (%s)  eval_every=%d (%s)  eval_mode=%s",
                int(sched.latest_every),
                str(sched.tick_unit),
                int(sched.eval_every),
                str(sched.tick_unit),
                str(train_cfg.eval.mode),
            )
        except Exception:
            pass

    bc_spec = BCTrainSpec(
        learning_rate=float(im.learning_rate),
        max_grad_norm=float(im.max_grad_norm),
        log_every_updates=max(1, int(sched.log_every)),
    )

    def _get_stat(stats: Dict[str, float], *keys: str) -> Optional[float]:
        for k in keys:
            v = stats.get(k, None)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def _maybe_tick_cores(*, phase: str) -> None:
        counter = _tick_counter(state, sched.tick_unit)

        latest_core.maybe_tick(progress_step=int(counter), model=model)

        def _extra_metrics() -> Dict[str, Any]:
            if not eval_sids:
                return {}

            val_seed = seed32_from(base_seed=int(run_cfg.seed), stream_id=0xBCE11 + int(counter))

            val_iter = iter_bc_batches_from_dataset(
                ds=ds,
                shard_ids=eval_sids,
                batch_size=int(im.batch_size),
                base_seed=int(val_seed),
                shuffle_shards=False,
                shuffle_within_shard=False,
                max_samples=0,
                drop_last=False,
                crop_top_rows=int(crop_top_rows),
                progress_cb=None,
                on_shard=None,
            )
            val_stats = bc_eval_stream(
                model=model,
                batch_iter=val_iter,
                device=device,
                max_samples=0,
            )

            out: Dict[str, Any] = {}
            if isinstance(val_stats, dict):
                for k, v in val_stats.items():
                    if not isinstance(k, str):
                        continue
                    kk = k
                    if kk.startswith("bc_val_"):
                        kk = "bc_val/" + kk[len("bc_val_") :]
                    out[kk] = v
            return out

        eval_core.maybe_tick(
            progress_step=int(counter),
            phase=str(phase),
            model=model,
            extra_metrics_fn=_extra_metrics if eval_sids else None,
        )

    # Pre-compute how many samples exist in train split (for sample-total bar).
    counts: dict[int, int] = {int(r.shard_id): int(r.num_samples) for r in ds.shards}
    train_total_samples_full = sum(int(counts.get(int(sid), 0)) for sid in train_sids)

    for e in range(max(1, int(im.epochs))):
        epoch_seed = seed32_from(base_seed=int(run_cfg.seed), stream_id=1009 * (e + 1))

        # If user caps max_samples, reflect it in the bar total.
        if int(im.max_samples) > 0:
            epoch_total_samples = min(int(im.max_samples), int(train_total_samples_full))
        else:
            epoch_total_samples = int(train_total_samples_full)
        epoch_total_samples_or_none = int(epoch_total_samples) if int(epoch_total_samples) > 0 else None

        with tqdm(
            total=epoch_total_samples_or_none,
            desc=f"bc epoch {int(e + 1)}/{int(im.epochs)}",
            unit="samp",
            leave=False,
            dynamic_ncols=True,
            position=0,
            mininterval=0.1,
        ) as epoch_bar, tqdm(
            total=int(len(train_sids)),
            desc="shards",
            unit="shard",
            leave=False,
            dynamic_ncols=True,
            position=2,
            mininterval=0.1,
        ) as shard_bar:

            def _on_shard(_sid: int) -> None:
                try:
                    shard_bar.update(1)
                except Exception:
                    pass

            batch_iter = iter_bc_batches_from_dataset(
                ds=ds,
                shard_ids=train_sids,
                batch_size=int(im.batch_size),
                base_seed=int(epoch_seed),
                shuffle_shards=bool(im.shuffle),
                shuffle_within_shard=True,
                max_samples=int(im.max_samples),
                drop_last=False,
                crop_top_rows=int(crop_top_rows),
                progress_cb=None,
                on_shard=_on_shard,
            )

            def _counting_iter() -> Any:
                nonlocal state
                for b in batch_iter:
                    if b is None:
                        continue

                    try:
                        act = b.get("action")
                        n = int(getattr(act, "shape", [0])[0])
                    except Exception:
                        n = 0

                    if n > 0:
                        state = ImitationRunState(samples_seen=int(state.samples_seen + n), updates=int(state.updates))
                        try:
                            epoch_bar.update(int(n))  # <-- samples, not batches
                        except Exception:
                            pass

                    yield b

            def _on_update(_u: int, stats: Dict[str, float]) -> None:
                nonlocal state
                state = ImitationRunState(samples_seen=int(state.samples_seen), updates=int(state.updates + 1))

                loss = _get_stat(stats, "bc/loss", "bc_loss", "loss")
                acc = _get_stat(stats, "bc/acc_top1", "bc_acc_top1", "acc_top1", "acc")
                ent = _get_stat(stats, "bc/entropy", "bc_entropy", "entropy")

                parts = []
                if loss is not None:
                    parts.append(f"loss={loss:.4g}")
                if acc is not None:
                    parts.append(f"acc={acc:.3f}")
                if ent is not None:
                    parts.append(f"H={ent:.3f}")

                try:
                    base = f"bc epoch {int(e + 1)}/{int(im.epochs)}"
                    if parts:
                        epoch_bar.set_description_str(base + "  " + "  ".join(parts), refresh=True)
                        epoch_bar.refresh()
                except Exception:
                    pass

                _maybe_tick_cores(phase="imitation")

            stats = bc_train_stream(
                model=model,
                batch_iter=_counting_iter(),
                spec=bc_spec,
                device=device,
                on_update=_on_update,
            )

        manager.save_latest(model=model, timesteps=int(_tick_counter(state, sched.tick_unit)))

        if logger is not None:
            try:
                logger.info(
                    "[imitation] epoch=%d/%d  samples_seen=%d  updates=%d  loss=%.6g",
                    int(e + 1),
                    int(im.epochs),
                    int(state.samples_seen),
                    int(state.updates),
                    float(stats.get("bc_loss", float("nan"))),
                )
            except Exception:
                pass

    return {
        "enabled": True,
        "dataset_dir": str(ds_dir),
        "train_shards": int(len(train_sids)),
        "eval_shards": int(len(eval_sids)),
        "samples_seen": int(state.samples_seen),
        "updates": int(state.updates),
    }


__all__ = ["ImitationRunnerSpec", "run_imitation"]
