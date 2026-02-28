# src/tetris_rl/core/callbacks/info_logger.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from tetris_rl.core.training.metrics import StatsAccumulator, StatsAccumulatorConfig


@dataclass(frozen=True)
class InfoLoggerSpec:
    """
    Logs per-step env metrics to TensorBoard using the stable info contract:

      info["tf"]   : transition features (reward-shaping + legality)
      info["game"] : game KPIs (score/level/lines totals etc.)
      info["ui"]   : watch-only payload (ignored by default)
    """

    log_every_calls: int = 200

    prefix_tf: str = "train/tf"
    prefix_game: str = "train/game"
    prefix_actions: str = "train/actions"

    verbose: int = 0

    log_action_histograms: bool = False
    hist_max_samples: int = 4096

    log_total_means: bool = False


class InfoLoggerCallback(BaseCallback):
    """
    TensorBoard logger for per-step env metrics.

    Reads the stable info contract:
      - info["tf"], info["game"] for metrics
      - optional info["ui"] for debug histograms

    This is TRAINING telemetry (rollout window). Real eval is handled elsewhere.
    """

    def __init__(
        self,
        *,
        spec: Optional[InfoLoggerSpec] = None,
        cfg: Optional[Dict[str, Any]] = None,
        log_every_calls: int = 50,
        prefix: str = "env",
        verbose: int = 0,
    ) -> None:
        if spec is None:
            base = str(prefix).strip().rstrip("/") if str(prefix).strip() else ""
            if base:
                prefix_tf = f"{base}/tf"
                prefix_game = f"{base}/game"
                prefix_actions = f"{base}/actions"
            else:
                prefix_tf = "train/tf"
                prefix_game = "train/game"
                prefix_actions = "train/actions"

            spec = InfoLoggerSpec(
                log_every_calls=int(log_every_calls),
                prefix_tf=prefix_tf,
                prefix_game=prefix_game,
                prefix_actions=prefix_actions,
                verbose=int(verbose),
            )

        super().__init__(verbose=int(spec.verbose))
        self.spec = spec
        self.cfg = cfg  # stored for consistency / future use
        self._calls = 0

        self._acc = StatsAccumulator(
            cfg=StatsAccumulatorConfig(
                log_action_histograms=bool(self.spec.log_action_histograms),
                hist_max_samples=int(self.spec.hist_max_samples),
                log_total_means=bool(self.spec.log_total_means),
            )
        )

    def _flush(self) -> None:
        if self._acc.steps <= 0:
            return

        summary = self._acc.summarize()
        for k, v in summary.items():
            if k.startswith("tf/"):
                kk = f"{str(self.spec.prefix_tf).rstrip('/')}/{k[len('tf/') :]}".rstrip("/")
            elif k.startswith("game/"):
                kk = f"{str(self.spec.prefix_game).rstrip('/')}/{k[len('game/') :]}".rstrip("/")
            else:
                kk = k
            self.logger.record(kk, float(v))

        if bool(self.spec.log_action_histograms):
            base = str(self.spec.prefix_actions).strip().rstrip("/")
            base = base if base else "train/actions"
            for hk, arr in self._acc.histograms().items():
                suffix = hk[len("actions/") :] if hk.startswith("actions/") else hk
                self.logger.record(f"{base}/{suffix}", arr)

        self._acc.reset()

    def _on_step(self) -> bool:
        self._calls += 1
        infos: Sequence[Mapping[str, Any]] = self.locals.get("infos", []) or []
        self._acc.ingest_infos(infos)

        if self.spec.log_every_calls > 0 and (self._calls % int(self.spec.log_every_calls)) == 0:
            self._flush()

        return True

    def _on_rollout_end(self) -> None:
        self._flush()

    def _on_training_end(self) -> None:
        self._flush()


__all__ = ["InfoLoggerSpec", "InfoLoggerCallback"]
