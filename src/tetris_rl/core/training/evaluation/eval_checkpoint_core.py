# src/tetris_rl/core/training/evaluation/eval_checkpoint_core.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.core.runs.checkpoints.checkpoint_manager import CheckpointManager, CheckpointPaths
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.training.config import EvalCheckpointCallbackConfig
from tetris_rl.core.training.evaluation import evaluate_model, evaluate_model_workers
from tetris_rl.core.training.evaluation.eval_metrics import as_float, pick_best_values, safe_int
from tetris_rl.core.training.evaluation.eval_table import EvalTable
from tetris_rl.core.training.evaluation.progress_ticker import ProgressTicker


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class EvalCheckpointCoreSpec:
    checkpoint_dir: Path
    eval_every: int
    run_cfg: RunConfig

    eval: EvalCheckpointCallbackConfig = field(default_factory=EvalCheckpointCallbackConfig)

    # injected by wiring code (cli/train.py)
    base_seed: int = 0

    # table output
    table_header_every: int = 10

    # progress metadata (purely for human clarity)
    progress_unit: str = "steps"  # "steps" | "samples" | "updates"

    verbose: int = 0


EvalRunner = Callable[
    [Any, int, Optional[Callable[[int, Optional[float]], None]], Optional[Callable[[int], None]]],
    Dict[str, Any],
]


class EvalCheckpointCore:
    """
    Framework-agnostic eval + best-checkpoint orchestrator.

    Call maybe_tick(...) from:
      - SB3 callback: progress_step=num_timesteps, phase="rl", unit="steps"
      - BC loop: progress_step=samples_seen, phase="imitation", unit="samples"
    """

    def __init__(
            self,
            *,
            spec: EvalCheckpointCoreSpec,
            cfg: Dict[str, Any],
            emit: Optional[Callable[[str], None]] = None,
            log_scalar: Optional[Callable[[str, float, int], None]] = None,
            eval_fn: Optional[EvalRunner] = None,
    ) -> None:
        self.spec = spec
        self.cfg = cfg
        self.emit = emit or (lambda s: print(s, flush=True))
        self.log_scalar = log_scalar

        self.eval_fn = eval_fn

        self.manager = CheckpointManager(
            paths=CheckpointPaths(checkpoint_dir=Path(spec.checkpoint_dir)),
            verbose=0,
        )
        self.ticker = ProgressTicker(every=int(spec.eval_every))
        self.table = EvalTable(
            emit=self.emit,
            table_header_every=int(spec.table_header_every),
            progress_unit=str(spec.progress_unit),
        )

    def init(self, *, progress_step: int) -> None:
        self.manager.ensure_dir()
        self.manager.load_state()
        self.ticker.init_from_progress(int(progress_step))
        if self.log_scalar is not None:
            step = int(progress_step)
            try:
                self.log_scalar("eval/every", float(self.spec.eval_every), step)
                self.log_scalar("eval/episodes_target", float(self.spec.eval.episodes), step)
                self.log_scalar("eval/min_steps", float(self.spec.eval.min_steps), step)
                if self.spec.eval.max_steps_per_episode is not None:
                    self.log_scalar(
                        "eval/max_steps_per_episode",
                        float(self.spec.eval.max_steps_per_episode),
                        step,
                    )
                self.log_scalar("eval/num_envs", float(self.spec.eval.num_envs), step)
                self.log_scalar("eval/workers", float(self.spec.eval.workers), step)
                mode = str(self.spec.eval.mode).strip().lower()
                self.log_scalar("eval/mode_vectorized", 1.0 if mode == "vectorized" else 0.0, step)
                self.log_scalar("eval/mode_workers", 1.0 if mode == "workers" else 0.0, step)
            except Exception:
                pass

    def _run_eval(self, *, model: Any, progress_step: int) -> Dict[str, Any]:
        seed_base = int(self.spec.base_seed) + int(self.spec.eval.seed_offset)

        eval_episodes = int(self.spec.eval.episodes)
        min_steps = int(self.spec.eval.min_steps)
        max_steps_per_episode = self.spec.eval.max_steps_per_episode
        deterministic = bool(self.spec.eval.deterministic)
        num_envs = max(1, int(self.spec.eval.num_envs))
        mode = str(self.spec.eval.mode).strip().lower()
        workers = max(1, int(self.spec.eval.workers))

        pbar: Any = None
        if int(self.spec.verbose) >= 1:
            pbar = tqdm(
                total=int(eval_episodes),
                desc=f"eval@{int(progress_step)}",
                leave=False,
                dynamic_ncols=True,
                position=1,
                unit="ep",
            )

        steps_seen = 0
        done_eps_seen = 0
        in_min_steps_drain = False

        def _refresh_postfix(ret: Optional[float] = None) -> None:
            if pbar is None:
                return
            if int(min_steps) > 0:
                suffix = " (collecting min_steps)" if in_min_steps_drain else ""
                msg = f"ep={done_eps_seen}/{eval_episodes} steps={steps_seen}/{min_steps}{suffix}"
            else:
                msg = f"ep={done_eps_seen}/{eval_episodes} steps={steps_seen}"
            if ret is not None:
                msg = f"{msg} return={float(ret):.6g}"
            pbar.set_postfix_str(msg, refresh=True)

        def _on_episode(done_eps: int, ret: Optional[float]) -> None:
            nonlocal done_eps_seen, in_min_steps_drain
            if pbar is None:
                return
            done_eps_seen = int(done_eps)
            if pbar.total is None:
                pbar.update(1)
            elif pbar.n < pbar.total:
                pbar.update(1)
            in_min_steps_drain = min_steps > 0 and done_eps_seen >= int(eval_episodes) and int(steps_seen) < int(min_steps)
            _refresh_postfix(ret)

        def _on_step(k: int) -> None:
            nonlocal steps_seen, in_min_steps_drain
            if pbar is None:
                return
            dk = int(k)
            if dk <= 0:
                return
            steps_seen += dk
            in_min_steps_drain = min_steps > 0 and done_eps_seen >= int(eval_episodes) and int(steps_seen) < int(min_steps)
            # Keep the UI alive when episode target is reached but step target is still not met.
            if in_min_steps_drain:
                if (int(steps_seen) % 256) == 0:
                    _refresh_postfix()
            elif done_eps_seen > 0 and ((int(steps_seen) % 2048) == 0):
                _refresh_postfix()

        try:
            if self.eval_fn is not None:
                metrics = self.eval_fn(
                    model,
                    int(progress_step),
                    _on_episode if pbar is not None else None,
                    _on_step if pbar is not None else None,
                )
            else:
                if mode == "workers":
                    metrics = evaluate_model_workers(
                        model=model,
                        cfg=self.cfg,  # wiring only
                        run_cfg=self.spec.run_cfg,
                        eval_episodes=eval_episodes,
                        min_steps=min_steps,
                        max_steps_per_episode=max_steps_per_episode,
                        deterministic=deterministic,
                        seed_base=seed_base,
                        workers=workers,
                        on_episode=_on_episode if pbar is not None else None,
                        on_step=_on_step if pbar is not None else None,
                    )
                elif mode == "vectorized":
                    metrics = evaluate_model(
                        model=model,
                        cfg=self.cfg,  # wiring only
                        run_cfg=self.spec.run_cfg,
                        eval_episodes=eval_episodes,
                        min_steps=min_steps,
                        max_steps_per_episode=max_steps_per_episode,
                        deterministic=deterministic,
                        seed_base=seed_base,
                        num_envs=num_envs,
                        on_episode=_on_episode if pbar is not None else None,
                        on_step=_on_step if pbar is not None else None,
                    )
                else:
                    raise ValueError(f"eval_checkpoint.mode must be 'vectorized' or 'workers' (got {mode!r})")
        finally:
            if pbar is not None:
                pbar.close()

        metrics["phase"] = "intermediate"
        metrics["progress_step"] = int(progress_step)
        metrics["progress_unit"] = str(self.spec.progress_unit)
        metrics["wall_time"] = _utc_now_iso()
        metrics["seed_base"] = int(seed_base)
        return metrics

    def maybe_tick(
            self,
            *,
            progress_step: int,
            phase: str,
            model: Any,
            extra_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> bool:
        if not self.ticker.should_tick(int(progress_step)):
            return False

        self.ticker.mark_ticked(int(progress_step))
        t = int(progress_step)

        self.table.emit_preamble(paths=self.manager.paths)

        metrics = self._run_eval(model=model, progress_step=t)

        # Optional extra metrics hook (e.g., offline BC validation on eval split).
        # This runs ONLY on ticks, and is fully owned by the caller (this core just merges the dict).
        if extra_metrics_fn is not None:
            try:
                extra = extra_metrics_fn()
                if isinstance(extra, dict) and extra:
                    metrics.update(extra)
            except Exception as e:
                # Keep eval robust: never crash training because an optional extra hook failed.
                try:
                    self.emit(f"[eval] WARN: extra_metrics_fn failed: {type(e).__name__}: {e}")
                except Exception:
                    pass

        self.manager.append_history(metrics)

        # optional scalar logging hook (SB3 uses it; BC can no-op)
        if self.log_scalar is not None:
            try:
                for k, v in metrics.items():
                    if isinstance(k, str) and k.startswith(("tf/", "game/", "episode/", "bc_val/")):
                        fv = as_float(v)
                        if fv is not None:
                            self.log_scalar(f"eval/{k}", float(fv), int(t))
            except Exception:
                pass

        reward_like, lines_like, survival_like = pick_best_values(metrics)

        did_r = did_n = did_t = False
        if reward_like is not None:
            did_r = self.manager.maybe_save_best(model=model, metric="reward", value=float(reward_like), timesteps=t)
        if lines_like is not None:
            did_n = self.manager.maybe_save_best(model=model, metric="lines", value=float(lines_like), timesteps=t)
        if survival_like is not None:
            did_t = self.manager.maybe_save_best(model=model, metric="survival", value=float(survival_like), timesteps=t)

        upd = ("R" if did_r else ".") + ("N" if did_n else ".") + ("T" if did_t else ".")

        ep_len = as_float(metrics.get("episode/steps_mean"))
        ep_ret = as_float(metrics.get("episode/return_mean"))
        rwd_per_step = None
        if ep_len is not None and ep_len > 0 and ep_ret is not None:
            rwd_per_step = float(ep_ret) / float(ep_len)
        if rwd_per_step is None:
            rwd_per_step = as_float(metrics.get("episode/return_per_step"))

        # Optional offline validation metrics (caller-defined).
        # Accept either namespaced keys or flat keys (be permissive at merge boundary).
        bc_vloss = as_float(metrics.get("bc_val/loss", metrics.get("bc_val_loss")))
        bc_vacc = as_float(metrics.get("bc_val/acc_top1", metrics.get("bc_val_acc_top1")))
        bc_vH = as_float(metrics.get("bc_val/entropy", metrics.get("bc_val_entropy")))
        weights = metrics.get("policy/weights")

        self.table.emit_row(
            t=t,
            phase=str(phase),
            upd=upd,
            steps=safe_int(metrics.get("eval/steps"), default=0),
            completed_eps=safe_int(metrics.get("episode/completed_episodes"), default=0),
            reward_per_step=rwd_per_step,
            lines_per_step=as_float(metrics.get("game/lines_per_step")),
            ep_len_mean=ep_len,
            invalid_action_rate=as_float(metrics.get("tf/invalid_action_rate")),
            weights=weights,
            bc_val_loss=bc_vloss,
            bc_val_acc_top1=bc_vacc,
            bc_val_entropy=bc_vH,
        )

        return True
