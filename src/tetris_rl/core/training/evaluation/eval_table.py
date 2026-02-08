# src/tetris_rl/core/training/evaluation/eval_table.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from tetris_rl.core.runs.checkpoints.checkpoint_manager import CheckpointPaths


class EvalTable:
    # fixed-width table layout; optional columns are omitted when unused
    _COL_W_T = 10
    _COL_W_PH = 10
    _COL_W_UPD = 6
    _COL_W_ST = 9
    _COL_W_EP = 5

    _COL_W_RPS = 10
    _COL_W_LPS = 10
    _COL_W_EPL = 8
    _COL_W_IAR = 8

    # optional extra columns (offline validation etc.)
    _COL_W_BCL = 10  # bc_val_loss
    _COL_W_BCA = 8  # bc_val_acc_top1
    _COL_W_BCH = 8  # bc_val_entropy

    def __init__(self, *, emit: Callable[[str], None], table_header_every: int, progress_unit: str) -> None:
        self.emit = emit
        self.table_header_every = int(table_header_every)
        self.progress_unit = str(progress_unit)
        self._tick_count = 0
        self._printed_preamble = False
        self._show_bc = False

    def _t_col_name(self) -> str:
        u = str(self.progress_unit).strip().lower()
        if u in {"samples", "sample"}:
            return "samples"
        if u in {"updates", "update"}:
            return "updates"
        if u in {"generation", "generations", "gen", "gens"}:
            return "gen"
        return "steps"

    def emit_preamble(self, *, paths: CheckpointPaths) -> None:
        if self._printed_preamble:
            return
        self._printed_preamble = True

        ckpt_dir = Path(paths.checkpoint_dir)

        self.emit(f"[eval] dir={ckpt_dir}  history={paths.history.name}")
        self.emit(
            "[eval] best checkpoints: "
            f"R={paths.best_reward.name}  N={paths.best_lines.name}  "
            f"T={paths.best_survival.name}"
        )
        self.emit(
            "[eval] update legend: "
            "R=best reward/step  N=best lines/step  T=best survival(1-go_rate)  .=no update"
        )
        self.emit(
            "[eval] cols: "
            f"{self._t_col_name()}=progress  algo=caller(rl|imitation|ga)  upd=best updates  "
            "steps=steps collected  ep=episodes finished  "
            "rwd/s=return_mean/steps_mean  ep_len=steps_mean (incl partial)  ill%=invalid_action_rate  "
        )
        self.emit("")

    def _table_header(self, *, show_bc: bool) -> str:
        tname = self._t_col_name()
        base = (
            f"{tname:>{self._COL_W_T}} "
            f"{'algo':<{self._COL_W_PH}} "
            f"{'upd':<{self._COL_W_UPD}} "
            f"{'steps':>{self._COL_W_ST}} "
            f"{'ep':>{self._COL_W_EP}} "
            f"{'rwd/s':>{self._COL_W_RPS}} "
            f"{'lines/s':>{self._COL_W_LPS}} "
            f"{'ep_len':>{self._COL_W_EPL}} "
            f"{'ill%':>{self._COL_W_IAR}} "
        )
        if not show_bc:
            return base
        return (
            base
            + f"{'bc_vloss':>{self._COL_W_BCL}} "
            + f"{'bc_acc':>{self._COL_W_BCA}} "
            + f"{'bc_H':>{self._COL_W_BCH}} "
        )

    def _table_sep(self, *, show_bc: bool) -> str:
        base = (
            f"{'-' * self._COL_W_T} "
            f"{'-' * self._COL_W_PH} "
            f"{'-' * self._COL_W_UPD} "
            f"{'-' * self._COL_W_ST} "
            f"{'-' * self._COL_W_EP} "
            f"{'-' * self._COL_W_RPS} "
            f"{'-' * self._COL_W_LPS} "
            f"{'-' * self._COL_W_EPL} "
            f"{'-' * self._COL_W_IAR} "
        )
        if not show_bc:
            return base
        return (
            base
            + f"{'-' * self._COL_W_BCL} "
            + f"{'-' * self._COL_W_BCA} "
            + f"{'-' * self._COL_W_BCH} "
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

    def emit_row(
        self,
        *,
        t: int,
        phase: str,
        upd: str,
        steps: Optional[int],
        completed_eps: Optional[int],
        reward_per_step: Optional[float],
        lines_per_step: Optional[float],
        ep_len_mean: Optional[float],
        invalid_action_rate: Optional[float],
        # optional extra metrics (offline validation etc.)
        bc_val_loss: Optional[float] = None,
        bc_val_acc_top1: Optional[float] = None,
        bc_val_entropy: Optional[float] = None,
    ) -> None:
        self._tick_count += 1
        show_bc = any(v is not None for v in (bc_val_loss, bc_val_acc_top1, bc_val_entropy))
        if show_bc and not self._show_bc:
            self._show_bc = True
            self.emit(self._table_header(show_bc=True))
            self.emit(self._table_sep(show_bc=True))
        elif self._tick_count == 1 or (
            self.table_header_every > 0 and (self._tick_count % int(self.table_header_every)) == 0
        ):
            self.emit(self._table_header(show_bc=self._show_bc))
            self.emit(self._table_sep(show_bc=self._show_bc))

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
            f"{self._fmt_float(lines_per_step, self._COL_W_LPS, prec=5)} "
            f"{self._fmt_float(ep_len_mean, self._COL_W_EPL, prec=1)} "
            f"{self._fmt_float(ill_pct, self._COL_W_IAR, prec=2)} "
        )
        if self._show_bc:
            row += (
                f"{self._fmt_float(bc_val_loss, self._COL_W_BCL, prec=4)} "
                f"{self._fmt_float(bc_val_acc_top1, self._COL_W_BCA, prec=3)} "
                f"{self._fmt_float(bc_val_entropy, self._COL_W_BCH, prec=3)} "
            )
        self.emit(row)


__all__ = ["EvalTable"]
