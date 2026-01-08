# src/tetris_rl/training/evaluation/eval_checkpoint_core.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.config.train_spec import TrainEvalSpec, TrainSpec
from tetris_rl.runs.checkpoint_manager import CheckpointManager, CheckpointPaths
from tetris_rl.training.evaluation import evaluate_model
from tetris_rl.training.evaluation.progress_ticker import ProgressTicker


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _mode_allows_phase(mode: str, phase: str) -> bool:
    m = str(mode).strip().lower()
    p = str(phase).strip().lower()
    if m == "off":
        return False
    if m == "both":
        return True
    return m == p


@dataclass(frozen=True)
class EvalCheckpointCoreSpec:
    checkpoint_dir: Path
    eval_every: int
    train_spec: TrainSpec

    eval: TrainEvalSpec = field(default_factory=TrainEvalSpec)

    # injected by wiring code (cli/train.py)
    base_seed: int = 0

    # table output
    table_header_every: int = 10

    # progress metadata (purely for human clarity)
    progress_unit: str = "steps"  # "steps" | "samples" | "updates"

    verbose: int = 0


class EvalCheckpointCore:
    """
    Framework-agnostic eval + best-checkpoint orchestrator.

    Call maybe_tick(...) from:
      - SB3 callback: progress_step=num_timesteps, phase="rl", unit="steps"
      - BC loop: progress_step=samples_seen, phase="imitation", unit="samples"
    """

    # fixed-width table layout (stable columns, no bouncing)
    _COL_W_T = 10
    _COL_W_PH = 10
    _COL_W_UPD = 6
    _COL_W_ST = 9
    _COL_W_EP = 5

    _COL_W_RPS = 10
    _COL_W_SPS = 10
    _COL_W_LPS = 10
    _COL_W_LVM = 8
    _COL_W_EPL = 8
    _COL_W_IAR = 8

    # optional extra columns (offline validation etc.)
    _COL_W_BCL = 10  # bc_val_loss
    _COL_W_BCA = 8  # bc_val_acc_top1
    _COL_W_BCH = 8  # bc_val_entropy

    def __init__(
            self,
            *,
            spec: EvalCheckpointCoreSpec,
            cfg: Dict[str, Any],
            emit: Optional[Callable[[str], None]] = None,
            log_scalar: Optional[Callable[[str, float, int], None]] = None,
    ) -> None:
        self.spec = spec
        self.cfg = cfg
        self.emit = emit or (lambda s: print(s, flush=True))
        self.log_scalar = log_scalar

        self.manager = CheckpointManager(
            paths=CheckpointPaths(checkpoint_dir=Path(spec.checkpoint_dir)),
            verbose=0,
        )
        self.ticker = ProgressTicker(every=int(spec.eval_every))

        self._tick_count: int = 0
        self._printed_preamble: bool = False

    def init(self, *, progress_step: int) -> None:
        self.manager.ensure_dir()
        self.manager.load_state()
        self.ticker.init_from_progress(int(progress_step))

    def _print_preamble_once(self) -> None:
        if self._printed_preamble:
            return
        self._printed_preamble = True

        p = self.manager.paths
        ckpt_dir = Path(p.checkpoint_dir)

        self.emit(f"[eval] dir={ckpt_dir}  history={p.history.name}")
        self.emit(
            "[eval] best checkpoints: "
            f"R={p.best_reward.name}  S={p.best_score.name}  N={p.best_lines.name}  V={p.best_level.name}  T={p.best_survival.name}"
        )
        self.emit(
            "[eval] update legend: "
            "R=best score/s  S=best final/mean score  N=best lines/s  V=best level_max  T=best survival(1-go_rate)  .=no update"
        )
        self.emit(
            "[eval] cols: "
            f"{self._t_col_name()}=progress  phase=caller(rl|imitation)  upd=best updates  steps=eval budget used  ep=episodes finished in budget  "
            "rwd/s=return_mean/steps_mean  ep_len=steps_mean (done=go or trunc)  ill%=invalid_action_rate  "
            "bc_val_*=optional offline validation metrics (if wired by caller)"
        )
        self.emit("")

    def _t_col_name(self) -> str:
        u = str(self.spec.progress_unit).strip().lower()
        if u in {"samples", "sample"}:
            return "samples"
        if u in {"updates", "update"}:
            return "updates"
        return "steps"

    def _table_header(self) -> str:
        tname = self._t_col_name()
        return (
            f"{tname:>{self._COL_W_T}} "
            f"{'phase':<{self._COL_W_PH}} "
            f"{'upd':<{self._COL_W_UPD}} "
            f"{'steps':>{self._COL_W_ST}} "
            f"{'ep':>{self._COL_W_EP}} "
            f"{'rwd/s':>{self._COL_W_RPS}} "
            f"{'score/s':>{self._COL_W_SPS}} "
            f"{'lines/s':>{self._COL_W_LPS}} "
            f"{'lv_max':>{self._COL_W_LVM}} "
            f"{'ep_len':>{self._COL_W_EPL}} "
            f"{'ill%':>{self._COL_W_IAR}} "
            f"{'bc_vloss':>{self._COL_W_BCL}} "
            f"{'bc_acc':>{self._COL_W_BCA}} "
            f"{'bc_H':>{self._COL_W_BCH}} "
        )

    def _table_sep(self) -> str:
        return (
            f"{'-' * self._COL_W_T} "
            f"{'-' * self._COL_W_PH} "
            f"{'-' * self._COL_W_UPD} "
            f"{'-' * self._COL_W_ST} "
            f"{'-' * self._COL_W_EP} "
            f"{'-' * self._COL_W_RPS} "
            f"{'-' * self._COL_W_SPS} "
            f"{'-' * self._COL_W_LPS} "
            f"{'-' * self._COL_W_LVM} "
            f"{'-' * self._COL_W_EPL} "
            f"{'-' * self._COL_W_IAR} "
            f"{'-' * self._COL_W_BCL} "
            f"{'-' * self._COL_W_BCA} "
            f"{'-' * self._COL_W_BCH} "
        )

    def _fmt_float(self, v: Optional[float], width: int, *, prec: int = 4) -> str:
        if v is None:
            return f"{'':>{width}}"
        s = f"{float(v):.{prec}f}" if abs(float(v)) < 1e6 else f"{float(v):.{prec}g}"
        if len(s) > width:
            s = s[:width]
        return f"{s:>{width}}"

    def _fmt_count(self, v: Optional[float], width: int) -> str:
        if v is None:
            return f"{'':>{width}}"

        n = float(v)
        if n >= 1_000_000:
            s = f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            s = f"{n / 1_000:.1f}k"
        else:
            s = f"{int(n)}"

        if len(s) > width:
            s = s[:width]
        return f"{s:>{width}}"

    def _fmt_int(self, v: Optional[int], width: int) -> str:
        if v is None:
            return f"{'':>{width}}"
        s = str(int(v))
        if len(s) > width:
            s = s[-width:]
        return f"{s:>{width}}"

    def _print_row(
            self,
            *,
            t: int,
            phase: str,
            upd: str,
            steps: Optional[int],
            completed_eps: Optional[int],
            reward_per_step: Optional[float],
            score_per_step: Optional[float],
            lines_per_step: Optional[float],
            level_max: Optional[float],
            ep_len_mean: Optional[float],
            invalid_action_rate: Optional[float],
            # optional extra metrics (offline validation etc.)
            bc_val_loss: Optional[float] = None,
            bc_val_acc_top1: Optional[float] = None,
            bc_val_entropy: Optional[float] = None,
    ) -> None:
        self._tick_count += 1

        if self._tick_count == 1 or (
                self.spec.table_header_every > 0 and (self._tick_count % int(self.spec.table_header_every)) == 0
        ):
            self.emit(self._table_header())
            self.emit(self._table_sep())

        upd_fixed = (upd[: self._COL_W_UPD]).ljust(self._COL_W_UPD)
        ill_pct = None if invalid_action_rate is None else 100.0 * float(invalid_action_rate)

        ph = str(phase).strip().lower()[: self._COL_W_PH].ljust(self._COL_W_PH)

        row = (
            f"{self._fmt_count(t, self._COL_W_T)} "
            f"{ph:<{self._COL_W_PH}} "
            f"{upd_fixed:<{self._COL_W_UPD}} "
            f"{self._fmt_count(steps, self._COL_W_ST)} "
            f"{self._fmt_int(completed_eps, self._COL_W_EP)} "
            f"{self._fmt_float(reward_per_step, self._COL_W_RPS, prec=5)} "
            f"{self._fmt_float(score_per_step, self._COL_W_SPS, prec=4)} "
            f"{self._fmt_float(lines_per_step, self._COL_W_LPS, prec=5)} "
            f"{self._fmt_float(level_max, self._COL_W_LVM, prec=2)} "
            f"{self._fmt_float(ep_len_mean, self._COL_W_EPL, prec=1)} "
            f"{self._fmt_float(ill_pct, self._COL_W_IAR, prec=2)} "
            f"{self._fmt_float(bc_val_loss, self._COL_W_BCL, prec=4)} "
            f"{self._fmt_float(bc_val_acc_top1, self._COL_W_BCA, prec=3)} "
            f"{self._fmt_float(bc_val_entropy, self._COL_W_BCH, prec=3)} "
        )
        self.emit(row)

    def _pick_best_values(
            self, metrics: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        score_per_step = _as_float(metrics.get("game/score_per_step"))
        lines_per_step = _as_float(metrics.get("game/lines_per_step"))
        level_max = _as_float(metrics.get("game/level_max"))

        score_like = _as_float(metrics.get("episode/final_score_mean"))
        if score_like is None:
            score_like = _as_float(metrics.get("game/score_mean"))

        go_rate = _as_float(metrics.get("tf/game_over_rate"))
        survival_like = None if go_rate is None else (1.0 - float(go_rate))

        reward_like = score_per_step
        return reward_like, score_like, lines_per_step, level_max, survival_like

    def _run_eval(self, *, model: Any, progress_step: int) -> Dict[str, Any]:
        seed_base = int(self.spec.base_seed) + int(self.spec.eval.seed_offset)

        eval_steps = int(self.spec.eval.steps)
        deterministic = bool(self.spec.eval.deterministic)
        num_envs = max(1, int(self.spec.eval.num_envs))

        pbar: Optional[tqdm] = None
        if int(self.spec.verbose) >= 1:
            pbar = tqdm(
                total=eval_steps,
                desc=f"eval@{int(progress_step)}",
                leave=False,
                dynamic_ncols=True,
                position=1,
            )

        def _on_episode(done_eps: int, ret: Optional[float]) -> None:
            if pbar is None:
                return
            if ret is not None:
                pbar.set_postfix_str(f"ep={done_eps} return={float(ret):.6g}", refresh=True)

        def _on_step(k: int) -> None:
            if pbar is None:
                return
            pbar.update(int(k))

        try:
            metrics = evaluate_model(
                model=model,
                cfg=self.cfg,  # wiring only
                train_spec=self.spec.train_spec,
                eval_steps=eval_steps,
                deterministic=deterministic,
                seed_base=seed_base,
                num_envs=num_envs,
                on_episode=_on_episode if pbar is not None else None,
                on_step=_on_step if pbar is not None else None,
            )
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
        # Gate intermediate eval by config mode, but do NOT print mode in the table (it's static config).
        mode = str(self.spec.eval.mode).strip().lower()
        if mode not in {"off", "rl", "imitation", "both"}:
            raise ValueError(f"TrainEvalSpec.mode must be off|rl|imitation|both (got {self.spec.eval.mode!r})")

        if not _mode_allows_phase(mode, str(phase)):
            return False

        if not self.ticker.should_tick(int(progress_step)):
            return False

        self.ticker.mark_ticked(int(progress_step))
        t = int(progress_step)

        self._print_preamble_once()

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
                        fv = _as_float(v)
                        if fv is not None:
                            self.log_scalar(f"eval/{k}", float(fv), int(t))
            except Exception:
                pass

        reward_like, score_like, lines_like, level_like, survival_like = self._pick_best_values(metrics)

        did_r = did_s = did_n = did_v = did_t = False
        if reward_like is not None:
            did_r = self.manager.maybe_save_best(model=model, metric="reward", value=float(reward_like), timesteps=t)
        if score_like is not None:
            did_s = self.manager.maybe_save_best(model=model, metric="score", value=float(score_like), timesteps=t)
        if lines_like is not None:
            did_n = self.manager.maybe_save_best(model=model, metric="lines", value=float(lines_like), timesteps=t)
        if level_like is not None:
            did_v = self.manager.maybe_save_best(model=model, metric="level", value=float(level_like), timesteps=t)
        if survival_like is not None:
            did_t = self.manager.maybe_save_best(model=model, metric="survival", value=float(survival_like), timesteps=t)

        upd = ("R" if did_r else ".") + ("S" if did_s else ".") + ("N" if did_n else ".") + ("V" if did_v else ".") + (
            "T" if did_t else "."
        )

        ep_len = _as_float(metrics.get("episode/steps_mean"))
        ep_ret = _as_float(metrics.get("episode/return_mean"))
        rwd_per_step = None
        if ep_len is not None and ep_len > 0 and ep_ret is not None:
            rwd_per_step = float(ep_ret) / float(ep_len)

        # Optional offline validation metrics (caller-defined).
        # Accept either namespaced keys or flat keys (be permissive at merge boundary).
        bc_vloss = _as_float(metrics.get("bc_val/loss", metrics.get("bc_val_loss")))
        bc_vacc = _as_float(metrics.get("bc_val/acc_top1", metrics.get("bc_val_acc_top1")))
        bc_vH = _as_float(metrics.get("bc_val/entropy", metrics.get("bc_val_entropy")))

        self._print_row(
            t=t,
            phase=str(phase),
            upd=upd,
            steps=_safe_int(metrics.get("eval/steps"), default=int(self.spec.eval.steps)),
            completed_eps=_safe_int(metrics.get("episode/completed_episodes"), default=0),
            reward_per_step=rwd_per_step,
            score_per_step=_as_float(metrics.get("game/score_per_step")),
            lines_per_step=_as_float(metrics.get("game/lines_per_step")),
            level_max=_as_float(metrics.get("game/level_max")),
            ep_len_mean=ep_len,
            invalid_action_rate=_as_float(metrics.get("tf/invalid_action_rate")),
            bc_val_loss=bc_vloss,
            bc_val_acc_top1=bc_vacc,
            bc_val_entropy=bc_vH,
        )

        return True
