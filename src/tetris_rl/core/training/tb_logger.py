# src/tetris_rl/core/training/tb_logger.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


class TensorboardLogger:
    def __init__(self, *, log_dir: Path) -> None:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard SummaryWriter is not available (torch not installed?)")
        self.log_dir = Path(log_dir)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(str(name), float(value), int(step))

    def log_dict(self, values: Dict[str, Any], *, step: int, prefix: str = "") -> None:
        base = str(prefix).strip().rstrip("/")
        for k, v in values.items():
            if not isinstance(k, str):
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            name = f"{base}/{k}" if base else k
            self._writer.add_scalar(name, fv, int(step))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def maybe_tb_logger(tb_dir: Optional[Path]) -> Optional[TensorboardLogger]:
    if tb_dir is None:
        return None
    try:
        return TensorboardLogger(log_dir=Path(tb_dir))
    except Exception:
        return None


__all__ = ["TensorboardLogger", "maybe_tb_logger"]
