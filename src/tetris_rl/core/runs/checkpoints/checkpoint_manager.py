# src/tetris_rl/core/runs/checkpoints/checkpoint_manager.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from tetris_rl.core.runs.checkpoints.checkpoint_manifest import CheckpointEntry, update_checkpoint_manifest
from tetris_rl.core.utils.file_io import append_jsonl, atomic_save_zip, read_json, write_json


@dataclass(frozen=True)
class CheckpointPaths:
    checkpoint_dir: Path

    @property
    def latest(self) -> Path:
        return self.checkpoint_dir / "latest.zip"

    @property
    def best_reward(self) -> Path:
        return self.checkpoint_dir / "best_reward.zip"

    @property
    def best_lines(self) -> Path:
        return self.checkpoint_dir / "best_lines.zip"

    @property
    def best_level(self) -> Path:
        return self.checkpoint_dir / "best_level.zip"

    @property
    def best_survival(self) -> Path:
        return self.checkpoint_dir / "best_survival.zip"

    @property
    def state(self) -> Path:
        return self.checkpoint_dir / "state.json"

    @property
    def history(self) -> Path:
        return self.checkpoint_dir / "eval_history.jsonl"


class CheckpointManager:
    """
    Owns checkpoint filenames and "best" bookkeeping.

    State format (state.json):
          {
        "latest": {"timesteps": int},
        "best": {
          "reward":   {"value": float, "timesteps": int},
          "lines":    {"value": float, "timesteps": int},
          "level":    {"value": float, "timesteps": int},
          "survival": {"value": float, "timesteps": int}
        }
      }
    """

    def __init__(self, *, paths: CheckpointPaths, verbose: int = 0) -> None:
        self.paths = paths
        self.verbose = int(verbose)
        self._state: Dict[str, Any] = {}

    def ensure_dir(self) -> None:
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> None:
        s = read_json(self.paths.state)
        if isinstance(s, dict):
            self._state = s
        else:
            self._state = {}

        if "best" not in self._state or not isinstance(self._state.get("best"), dict):
            self._state["best"] = {}
        else:
            self._state["best"].pop("score", None)

    def _write_state(self) -> None:
        write_json(self.paths.state, self._state)

    def append_history(self, record: Dict[str, Any]) -> None:
        append_jsonl(self.paths.history, record)

    def save_latest(self, *, model, timesteps: int) -> bool:
        atomic_save_zip(model=model, dst=self.paths.latest)
        self._state.setdefault("latest", {})
        self._state["latest"]["timesteps"] = int(timesteps)
        self._write_state()
        update_checkpoint_manifest(
            manifest_path=self.paths.checkpoint_dir / "manifest.json",
            field="latest",
            entry=CheckpointEntry(
                path=self.paths.latest.name,
                timesteps=int(timesteps),
            ),
        )
        return True

    def maybe_save_best(self, *, model, metric: str, value: float, timesteps: int) -> bool:
        metric = str(metric).strip().lower()

        # aliases / backward-compat knobs
        if metric in {"len", "length", "time"}:
            metric = "survival"

        dst = self._metric_to_path(metric)
        if dst is None:
            raise KeyError(f"unknown best metric {metric!r}")

        best = self._state.setdefault("best", {})
        cur = best.get(metric)
        cur_v: Optional[float] = None
        if isinstance(cur, dict) and "value" in cur:
            try:
                cur_v = float(cur["value"])
            except Exception:
                cur_v = None

        v = float(value)
        if cur_v is not None and v <= cur_v:
            return False

        # save and update state
        atomic_save_zip(model=model, dst=dst)
        best[metric] = {"value": float(v), "timesteps": int(timesteps)}
        self._write_state()
        manifest_field = self._metric_to_manifest_field(metric)
        if manifest_field is not None:
            update_checkpoint_manifest(
                manifest_path=self.paths.checkpoint_dir / "manifest.json",
                field=manifest_field,
                entry=CheckpointEntry(
                    path=dst.name,
                    timesteps=int(timesteps),
                    metric=str(metric),
                    value=float(v),
                ),
            )
        return True

    def _metric_to_path(self, metric: str) -> Optional[Path]:
        if metric == "reward":
            return self.paths.best_reward
        if metric == "lines":
            return self.paths.best_lines
        if metric == "level":
            return self.paths.best_level
        if metric == "survival":
            return self.paths.best_survival
        return None

    def _metric_to_manifest_field(self, metric: str) -> Optional[str]:
        if metric == "reward":
            return "best_reward"
        if metric == "lines":
            return "best_lines"
        if metric == "level":
            return "best_level"
        if metric == "survival":
            return "best_survival"
        return None
