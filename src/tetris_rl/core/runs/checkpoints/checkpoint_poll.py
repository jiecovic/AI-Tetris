# src/tetris_rl/core/runs/checkpoints/checkpoint_poll.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from tetris_rl.core.runs.checkpoints.checkpoint_manifest import resolve_checkpoint_from_manifest
from tetris_rl.core.training.config import AlgoConfig
from tetris_rl.core.training.model_io import load_model_from_algo_config


def _safe_mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


@dataclass
class CheckpointPoller:
    """
    Polls a run's checkpoints and reloads the model if the selected checkpoint updates.

    - Uses mtime to detect changes.
    - Does NOT reset env/game on reload (watch is continuous).
    - Loads via load_model_from_algo_config(algo_cfg=...) so PPO vs MaskablePPO is handled correctly.
    """
    run_dir: Path
    which: str
    algo_cfg: AlgoConfig
    device: str = "auto"
    reload_every_s: float = 1.0
    # Optional custom loader used by runtimes with non-SB3 checkpoint formats
    # (e.g., imitation policy-only checkpoints). Signature: (ckpt_path, device)->(model, algo_type).
    loader: Optional[Callable[[Path, str], tuple[Any, str]]] = None

    _ckpt: Optional[Path] = None
    _mtime: float = 0.0
    _model: Optional[Any] = None
    _algo_type: str = "ppo"
    _last_poll_s: float = 0.0
    reload_count: int = 0

    def set_current(self, *, ckpt: Path, model: Any, algo_type: str) -> None:
        self._ckpt = Path(ckpt)
        self._mtime = _safe_mtime(self._ckpt)
        self._model = model
        self._algo_type = str(algo_type).strip().lower() if algo_type is not None else "ppo"

    def maybe_reload(self, *, now_s: Optional[float] = None) -> Optional[Tuple[Any, Path, str]]:
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

        # unchanged file -> nothing to do
        if cur is not None and chosen == cur and mtime <= cur_m:
            return None

        try:
            if self.loader is not None:
                model, algo_type = self.loader(chosen, str(self.device))
                loaded_model = model
                loaded_algo_type = str(algo_type).strip().lower()
                loaded_ckpt = chosen
            else:
                loaded = load_model_from_algo_config(algo_cfg=self.algo_cfg, ckpt=chosen, device=str(self.device))
                loaded_model = loaded.model
                loaded_algo_type = str(getattr(loaded, "algo_type", "ppo")).strip().lower()
                loaded_ckpt = loaded.ckpt
        except Exception:
            # Keep watch loop alive if a reload attempt fails (e.g. transient file state).
            return None

        self._ckpt = loaded_ckpt
        self._mtime = _safe_mtime(self._ckpt)
        self._model = loaded_model
        self._algo_type = loaded_algo_type

        self.reload_count += 1
        return self._model, self._ckpt, self._algo_type

    @property
    def current_ckpt(self) -> Optional[Path]:
        return self._ckpt

    @property
    def algo_type(self) -> str:
        return str(self._algo_type)
