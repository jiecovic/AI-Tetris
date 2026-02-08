# src/tetris_rl/core/training/imitation/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm.rich import tqdm  # <-- rich tqdm (NOT tqdm.auto)

from planning_rl.utils.seed import seed32_from
from tetris_rl.core.callbacks import CallbackList, EvalCallback, LatestCallback
from tetris_rl.core.datagen.io.shard_reader import ShardDataset
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.training.config import CallbacksConfig, ImitationLearnConfig
from tetris_rl.core.training.evaluation import evaluate_model
from tetris_rl.core.training.evaluation.eval_checkpoint_core import EvalCheckpointCoreSpec
from tetris_rl.core.training.evaluation.latest_checkpoint_core import LatestCheckpointCoreSpec
from tetris_rl.core.training.imitation.algorithm import ImitationAlgorithm
from tetris_rl.core.training.imitation.bc import BCTrainSpec, bc_eval_stream, bc_train_stream
from tetris_rl.core.training.imitation.data import iter_bc_batches_from_dataset, split_shards_modulo
from tetris_rl.core.training.imitation.types import ImitationRunState, ImitationScheduleSpec, ImitationSplitSpec
from tetris_rl.core.training.tb_logger import TensorboardLogger
from tetris_rl.core.utils.paths import repo_root


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


class ImitationTrainer:
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        algo: ImitationAlgorithm,
        learn_cfg: ImitationLearnConfig,
        callbacks_cfg: CallbacksConfig,
        run_cfg: RunConfig,
        run_dir: Path,
        repo: Optional[Path] = None,
        logger: Any = None,
        tb_logger: Optional[TensorboardLogger] = None,
    ) -> None:
        self.cfg = cfg
        self.algo = algo
        self.learn_cfg = learn_cfg
        self.callbacks_cfg = callbacks_cfg
        self.run_cfg = run_cfg
        self.run_dir = Path(run_dir)
        self.repo = Path(repo) if repo is not None else repo_root()
        self.logger = logger
        self.tb_logger = tb_logger

    def run(self) -> Dict[str, Any]:
        learn = self.learn_cfg

        ds_dir = _resolve_dataset_dir(self.repo, str(learn.dataset_dir))
        ds = ShardDataset(dataset_dir=ds_dir)

        ds_h = int(ds.manifest.board_h)
        ds_w = int(ds.manifest.board_w)
        obs_h, obs_w = _infer_policy_grid_hw(self.algo)
        crop_top_rows = _compute_dataset_grid_crop(
            ds_h=ds_h,
            ds_w=ds_w,
            obs_h=obs_h,
            obs_w=obs_w,
            dataset_dir=ds_dir,
        )

        if self.logger is not None:
            try:
                if crop_top_rows > 0:
                    self.logger.info(
                        "[imitation] grid adapter: dataset grid=(%d,%d) -> policy grid=(%d,%d) by cropping top_rows=%d",
                        ds_h,
                        ds_w,
                        obs_h,
                        obs_w,
                        crop_top_rows,
                    )
                else:
                    self.logger.info("[imitation] grid adapter: dataset grid matches policy grid (%d,%d)", obs_h, obs_w)
            except Exception:
                pass

        shard_ids = [int(s.shard_id) for s in ds.shards]
        split = split_shards_modulo(
            shard_ids=shard_ids,
            base_seed=int(self.run_cfg.seed),
            eval_mod=200,
            eval_mod_offset=0,
            seed_offset=12345,
        )
        train_sids = split.train
        eval_sids = split.eval

        eval_every = (
            int(self.callbacks_cfg.eval_checkpoint.every)
            if bool(self.callbacks_cfg.eval_checkpoint.enabled)
            else 0
        )
        latest_every = int(self.callbacks_cfg.latest.every) if bool(self.callbacks_cfg.latest.enabled) else 0

        sched = ImitationScheduleSpec(
            tick_unit="samples",
            latest_every=latest_every,
            eval_every=eval_every,
            log_every=50,
        )

        state = ImitationRunState(samples_seen=0, updates=0)
        device = str(self.run_cfg.device).strip() if str(self.run_cfg.device).strip() else "cpu"
        progress_event = "update" if str(sched.tick_unit) == "updates" else "sample"
        progress_key = "updates" if progress_event == "update" else "samples_seen"

        callback_items = []
        if int(sched.latest_every) > 0:
            callback_items.append(
                LatestCallback(
                    spec=LatestCheckpointCoreSpec(
                        checkpoint_dir=Path(self.run_dir) / "checkpoints",
                        latest_every=int(sched.latest_every),
                        verbose=0,
                    ),
                    event=progress_event,
                    progress_key=progress_key,
                    model_getter=(lambda algo: algo),
                )
            )

        def _extra_bc_metrics() -> Dict[str, Any]:
            if not eval_sids:
                return {}

            val_seed = seed32_from(base_seed=int(self.run_cfg.seed), stream_id=0xBCE11 + int(state.samples_seen))

            val_iter = iter_bc_batches_from_dataset(
                ds=ds,
                shard_ids=eval_sids,
                batch_size=int(learn.batch_size),
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
                model=self.algo,
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

        if int(sched.eval_every) > 0:
            eval_seed_base = int(self.run_cfg.seed) + int(self.callbacks_cfg.eval_checkpoint.seed_offset)

            def _eval_fn(_model: Any, _t: int, on_episode, on_step) -> Dict[str, Any]:
                metrics = evaluate_model(
                    model=_model,
                    cfg=self.cfg,
                    run_cfg=self.run_cfg,
                    eval_episodes=int(self.callbacks_cfg.eval_checkpoint.episodes),
                    min_steps=int(self.callbacks_cfg.eval_checkpoint.min_steps),
                    max_steps_per_episode=self.callbacks_cfg.eval_checkpoint.max_steps_per_episode,
                    deterministic=bool(self.callbacks_cfg.eval_checkpoint.deterministic),
                    seed_base=int(eval_seed_base),
                    num_envs=int(self.callbacks_cfg.eval_checkpoint.num_envs),
                    on_episode=on_episode,
                    on_step=on_step,
                )
                metrics.update(_extra_bc_metrics())
                return metrics

            callback_items.append(
                EvalCallback(
                    spec=EvalCheckpointCoreSpec(
                        checkpoint_dir=Path(self.run_dir) / "checkpoints",
                        eval_every=int(sched.eval_every),
                        run_cfg=self.run_cfg,
                        eval=self.callbacks_cfg.eval_checkpoint,
                        base_seed=int(self.run_cfg.seed),
                        table_header_every=10,
                        progress_unit=str(sched.tick_unit),
                        verbose=1,
                    ),
                    cfg=self.cfg,
                    event=progress_event,
                    progress_key=progress_key,
                    phase="imitation",
                    model_getter=(lambda algo: algo),
                    emit=(lambda s: self.logger.info(s)) if self.logger is not None else None,
                    log_scalar=(self.tb_logger.log_scalar if self.tb_logger is not None else None),
                    eval_fn=_eval_fn,
                )
            )

        callbacks = CallbackList(callback_items) if callback_items else None
        if callbacks is not None:
            callbacks.init_callback(self.algo)
            callbacks.on_start(samples_seen=int(state.samples_seen), updates=int(state.updates))

        if self.logger is not None:
            try:
                self.logger.info(
                    "[imitation] dataset=%s  shards(train=%d eval=%d)  batch=%d  epochs=%d  max_samples=%d",
                    str(ds_dir),
                    int(len(train_sids)),
                    int(len(eval_sids)),
                    int(learn.batch_size),
                    int(learn.epochs),
                    int(learn.max_samples),
                )
                self.logger.info(
                    "[imitation] cadences: latest_every=%d (%s)  eval_every=%d (%s)  eval_enabled=%s",
                    int(sched.latest_every),
                    str(sched.tick_unit),
                    int(sched.eval_every),
                    str(sched.tick_unit),
                    str(self.callbacks_cfg.eval_checkpoint.enabled),
                )
            except Exception:
                pass

        bc_spec = BCTrainSpec(
            learning_rate=float(learn.learning_rate),
            max_grad_norm=float(learn.max_grad_norm),
            log_every_updates=max(1, int(sched.log_every)),
        )

        def _get_stat(stats: Dict[str, float], *keys: str) -> Optional[float]:
            for k in keys:
                v = stats.get(k, None)
                if isinstance(v, (int, float)):
                    return float(v)
            return None

        counts: dict[int, int] = {int(r.shard_id): int(r.num_samples) for r in ds.shards}
        train_total_samples_full = sum(int(counts.get(int(sid), 0)) for sid in train_sids)

        for e in range(max(1, int(learn.epochs))):
            epoch_seed = seed32_from(base_seed=int(self.run_cfg.seed), stream_id=1009 * (e + 1))

            if int(learn.max_samples) > 0:
                epoch_total_samples = min(int(learn.max_samples), int(train_total_samples_full))
            else:
                epoch_total_samples = int(train_total_samples_full)
            epoch_total_samples_or_none = int(epoch_total_samples) if int(epoch_total_samples) > 0 else None

            with tqdm(
                total=epoch_total_samples_or_none,
                desc=f"bc epoch {int(e + 1)}/{int(learn.epochs)}",
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
                    batch_size=int(learn.batch_size),
                    base_seed=int(epoch_seed),
                    shuffle_shards=bool(learn.shuffle),
                    shuffle_within_shard=True,
                    max_samples=int(learn.max_samples),
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
                            if callbacks is not None and progress_event == "sample":
                                callbacks.on_event(
                                    event=progress_event,
                                    samples_seen=int(state.samples_seen),
                                    updates=int(state.updates),
                                )

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
                        base = f"bc epoch {int(e + 1)}/{int(learn.epochs)}"
                        if parts:
                            epoch_bar.set_description_str(base + "  " + "  ".join(parts), refresh=True)
                            epoch_bar.refresh()
                    except Exception:
                        pass

                    if callbacks is not None and progress_event == "update":
                        callbacks.on_event(
                            event=progress_event,
                            samples_seen=int(state.samples_seen),
                            updates=int(state.updates),
                        )

                    if self.tb_logger is not None:
                        step = int(state.samples_seen)
                        if loss is not None:
                            self.tb_logger.log_scalar("train/bc_loss", float(loss), step)
                        if acc is not None:
                            self.tb_logger.log_scalar("train/bc_acc_top1", float(acc), step)
                        if ent is not None:
                            self.tb_logger.log_scalar("train/bc_entropy", float(ent), step)

                stats = bc_train_stream(
                    model=self.algo,
                    batch_iter=_counting_iter(),
                    spec=bc_spec,
                    device=device,
                    on_update=_on_update,
                )

            if self.logger is not None:
                try:
                    self.logger.info(
                        "[imitation] epoch=%d/%d  samples_seen=%d  updates=%d  loss=%.6g",
                        int(e + 1),
                        int(learn.epochs),
                        int(state.samples_seen),
                        int(state.updates),
                        float(stats.get("bc_loss", float("nan"))),
                    )
                except Exception:
                    pass

        if callbacks is not None:
            callbacks.on_end(samples_seen=int(state.samples_seen), updates=int(state.updates))

        if self.tb_logger is not None:
            self.tb_logger.flush()

        return {
            "dataset_dir": str(ds_dir),
            "train_shards": int(len(train_sids)),
            "eval_shards": int(len(eval_sids)),
            "samples_seen": int(state.samples_seen),
            "updates": int(state.updates),
        }


__all__ = ["ImitationRunnerSpec", "ImitationTrainer"]
