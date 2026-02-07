# src/tetris_rl/core/training/callbacks/latest_checkpoint.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback

try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.core.training.evaluation.latest_checkpoint_core import LatestCheckpointCore, LatestCheckpointCoreSpec


@dataclass(frozen=True)
class LatestCheckpointSpec:
    checkpoint_dir: Path
    latest_every: int = 50_000
    verbose: int = 0


class LatestCheckpointCallback(BaseCallback):
    """
    SB3 adapter: saves checkpoints/latest.zip on its own cadence.
    """

    def __init__(self, *, spec: LatestCheckpointSpec, cfg: Dict[str, Any]) -> None:
        super().__init__(verbose=int(spec.verbose))
        self.spec = spec
        self.cfg = cfg

        def _emit(line: str) -> None:
            try:
                tqdm.write(line)
            except Exception:
                print(line, flush=True)

        self.core = LatestCheckpointCore(
            spec=LatestCheckpointCoreSpec(
                checkpoint_dir=Path(spec.checkpoint_dir),
                latest_every=int(spec.latest_every),
                verbose=int(spec.verbose),
            ),
            emit=_emit,
        )

    def _init_callback(self) -> None:
        self.core.init(progress_step=int(self.num_timesteps))

    def _on_step(self) -> bool:
        self.core.maybe_tick(progress_step=int(self.num_timesteps), model=self.model)
        return True


__all__ = ["LatestCheckpointSpec", "LatestCheckpointCallback"]
