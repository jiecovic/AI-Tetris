# src/tetris_rl/rewardfit/models_old/adapters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np

from tetris_rl.rewardfit.types import FitResult


class FitFn(Protocol):
    def __call__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            logger: Any,
            which: str,
            feature_names: Sequence[str],
            fit_intercept: bool = True,
    ) -> list[FitResult]:
        ...


class PredictFn(Protocol):
    def __call__(self, X: np.ndarray, fit: FitResult, base_feature_dim: int) -> np.ndarray:
        ...


class ScaleExtraFn(Protocol):
    def __call__(self, extra: Dict[str, object], scale: float) -> Dict[str, object]:
        ...


class SaveExtraFn(Protocol):
    def __call__(self, save_dict: Dict[str, object], best_raw: FitResult, best_scaled: FitResult) -> None:
        ...


@dataclass(frozen=True)
class ModelAdapter:
    """
    Model-specific hooks. Most models_old only need `fit`.

    predict:
      - used only for normalize=std
      - must return r(X) in the model's native parameterization

    scale_extra:
      - keep fit.extra consistent with scaled coef/bias for inference/export

    save_extra:
      - populate np.savez payload with model-specific fields
    """
    kind: str
    fit: FitFn
    predict: Optional[PredictFn] = None
    scale_extra: Optional[ScaleExtraFn] = None
    save_extra: Optional[SaveExtraFn] = None
