# src/tetris_rl/training/callbacks/eval_checkpoint.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback

try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.runs.config import RunConfig
from tetris_rl.training.config import TrainEvalConfig
from tetris_rl.training.evaluation.eval_checkpoint_core import EvalCheckpointCore, EvalCheckpointCoreSpec


@dataclass(frozen=True)
class EvalCheckpointSpec:
    checkpoint_dir: Path
    eval_every: int
    run_cfg: RunConfig

    eval: TrainEvalConfig = field(default_factory=TrainEvalConfig)
    verbose: int = 0

    # injected by wiring code (cli/train.py)
    base_seed: int = 0

    table_header_every: int = 10


class EvalCheckpointCallback(BaseCallback):
    """
    SB3 adapter: eval + best-checkpoint orchestration (step-budget evaluation).
    """

    def __init__(self, *, spec: EvalCheckpointSpec, cfg: Dict[str, Any]) -> None:
        super().__init__(verbose=int(spec.verbose))
        self.spec = spec
        self.cfg = cfg

        def _emit(line: str) -> None:
            try:
                tqdm.write(line)
            except Exception:
                print(line, flush=True)

        def _log_scalar(name: str, value: float, step: int) -> None:
            try:
                self.logger.record(str(name), float(value))
            except Exception:
                pass

        self.core = EvalCheckpointCore(
            spec=EvalCheckpointCoreSpec(
                checkpoint_dir=Path(spec.checkpoint_dir),
                eval_every=int(spec.eval_every),
                run_cfg=spec.run_cfg,
                eval=spec.eval,
                base_seed=int(spec.base_seed),
                table_header_every=int(spec.table_header_every),
                progress_unit="steps",
                verbose=max(1, int(spec.verbose)),
            ),
            cfg=cfg,
            emit=_emit,
            log_scalar=_log_scalar,
        )

    def _init_callback(self) -> None:
        self.core.init(progress_step=int(self.num_timesteps))

    def _on_step(self) -> bool:
        # this callback runs in SB3 learn() -> phase is RL progress
        self.core.maybe_tick(progress_step=int(self.num_timesteps), phase="rl", model=self.model)
        return True


__all__ = ["EvalCheckpointSpec", "EvalCheckpointCallback"]
