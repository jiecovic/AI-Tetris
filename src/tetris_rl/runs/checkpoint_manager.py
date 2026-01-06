# src/tetris_rl/runs/checkpoint_manager.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from tetris_rl.utils.file_io import append_jsonl, atomic_save_zip, read_json, write_json


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
    def best_score(self) -> Path:
        return self.checkpoint_dir / "best_score.zip"

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


def resolve_checkpoint_path(paths: CheckpointPaths, which: str) -> Path:
    """
    Resolve a user-facing checkpoint selector into a concrete checkpoint path.

    Supported:
      - latest
      - best (alias for best_reward)
      - reward
      - score
      - lines
      - level
      - survival (alias: len, length, time)

    Notes:
      - 'final' is not produced by the checkpoint manager; callers may special-case it.
      - If a best_* file does not exist, callers may fall back to latest.
    """
    w = str(which).strip().lower()
    if w in {"latest"}:
        return paths.latest
    if w in {"best", "reward"}:
        return paths.best_reward
    if w in {"score"}:
        return paths.best_score
    if w in {"lines"}:
        return paths.best_lines
    if w in {"level"}:
        return paths.best_level
    if w in {"survival", "len", "length", "time"}:
        return paths.best_survival
    raise ValueError(f"unknown checkpoint selector: {which!r}")


def resolve_checkpoint_selector(*, run_dir: Path, which: str) -> Path:
    """
    Resolve a checkpoint selector into an on-disk path.

    - Supports: latest, best/reward, score, lines, level, survival, final
    - Applies fallback: best/reward -> latest if missing
    - Returns the resolved path even if it doesn't exist (callers may check is_file()).
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    paths = CheckpointPaths(checkpoint_dir=ckpt_dir)

    w = str(which).strip().lower()
    if w == "final":
        return ckpt_dir / "final.zip"

    p = resolve_checkpoint_path(paths, w)
    if p.is_file():
        return p

    if w in {"best", "reward"} and paths.latest.is_file():
        return paths.latest

    return p


class CheckpointManager:
    """
    Owns checkpoint filenames and "best" bookkeeping.

    State format (state.json):
      {
        "latest": {"timesteps": int},
        "best": {
          "reward":   {"value": float, "timesteps": int},
          "score":    {"value": float, "timesteps": int},
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

    def _write_state(self) -> None:
        write_json(self.paths.state, self._state)

    def append_history(self, record: Dict[str, Any]) -> None:
        append_jsonl(self.paths.history, record)

    def save_latest(self, *, model, timesteps: int) -> bool:
        atomic_save_zip(model=model, dst=self.paths.latest)
        self._state.setdefault("latest", {})
        self._state["latest"]["timesteps"] = int(timesteps)
        self._write_state()
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
        return True

    def _metric_to_path(self, metric: str) -> Optional[Path]:
        if metric == "reward":
            return self.paths.best_reward
        if metric == "score":
            return self.paths.best_score
        if metric == "lines":
            return self.paths.best_lines
        if metric == "level":
            return self.paths.best_level
        if metric == "survival":
            return self.paths.best_survival
        return None
