# src/tetris_rl/core/runs/checkpoints/planning_poll.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from tetris_rl.core.runs.checkpoints.checkpoint_manifest import resolve_checkpoint_from_manifest


def _safe_mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


@dataclass
class PlanningCheckpointPoller:
    """
    Polls planning checkpoints (TD/GA) and reloads a policy when the file updates.
    """

    run_dir: Path
    which: str
    reload_every_s: float
    loader: Callable[[Path], Any]

    _ckpt: Optional[Path] = None
    _mtime: float = 0.0
    _policy: Optional[Any] = None
    _last_poll_s: float = 0.0
    reload_count: int = 0

    def set_current(self, *, ckpt: Path, policy: Any) -> None:
        self._ckpt = Path(ckpt)
        self._mtime = _safe_mtime(self._ckpt)
        self._policy = policy

    def maybe_reload(self, *, now_s: Optional[float] = None) -> Optional[Tuple[Any, Path]]:
        if float(self.reload_every_s) <= 0:
            return None

        now = float(time.time() if now_s is None else now_s)
        if (now - float(self._last_poll_s)) < float(self.reload_every_s):
            return None
        self._last_poll_s = now

        try:
            chosen = resolve_checkpoint_from_manifest(run_dir=self.run_dir, which=self.which)
        except Exception:
            return None
        if not chosen.is_file():
            return None

        mtime = _safe_mtime(chosen)
        cur = self._ckpt
        cur_m = float(self._mtime)

        if cur is not None and chosen == cur and mtime <= cur_m:
            return None

        policy = self.loader(chosen)
        self._ckpt = chosen
        self._mtime = _safe_mtime(chosen)
        self._policy = policy
        self.reload_count += 1
        return policy, chosen

    @property
    def current_ckpt(self) -> Optional[Path]:
        return self._ckpt


__all__ = ["PlanningCheckpointPoller"]
