# src/tetris_rl/core/training/evaluation/latest_checkpoint_core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from tetris_rl.core.runs.checkpoints.checkpoint_manager import CheckpointManager, CheckpointPaths
from tetris_rl.core.training.evaluation.progress_ticker import ProgressTicker


@dataclass(frozen=True)
class LatestCheckpointCoreSpec:
    checkpoint_dir: Path
    latest_every: int = 50_000

    # informational output
    verbose: int = 0


class LatestCheckpointCore:
    """
    Framework-agnostic latest checkpoint saver.

    Call maybe_tick(progress_step=...) from:
      - SB3 callback (progress_step=num_timesteps)
      - BC loop (progress_step=samples_seen or updates)
    """

    def __init__(
            self,
            *,
            spec: LatestCheckpointCoreSpec,
            emit: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.spec = spec
        self.emit = emit

        self.manager = CheckpointManager(
            paths=CheckpointPaths(checkpoint_dir=Path(spec.checkpoint_dir)),
            verbose=0,
        )
        self.ticker = ProgressTicker(every=int(spec.latest_every))
        self._printed = False

    def init(self, *, progress_step: int) -> None:
        self.manager.ensure_dir()
        self.manager.load_state()
        self.ticker.init_from_progress(int(progress_step))

    def maybe_tick(self, *, progress_step: int, model: Any) -> bool:
        if not self.ticker.should_tick(int(progress_step)):
            return False

        self.ticker.mark_ticked(int(progress_step))
        t = int(progress_step)

        self.manager.save_latest(model=model, timesteps=t)

        if int(self.spec.verbose) >= 1 and (not self._printed) and self.emit is not None:
            self._printed = True
            self.emit(f"[latest] saving {self.manager.paths.latest.name} every {int(self.spec.latest_every)} steps")

        return True
