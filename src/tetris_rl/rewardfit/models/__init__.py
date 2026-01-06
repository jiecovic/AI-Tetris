# src/tetris_rl/rewardfit/models_old/__init__.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from tetris_rl.rewardfit.models.adapters import ModelAdapter
from tetris_rl.rewardfit.models.isotonic_gam import make_isotonic_adapter
from tetris_rl.rewardfit.models.linear_family import make_linear_adapters
from tetris_rl.rewardfit.models.mlp import make_mlp_adapter
from tetris_rl.rewardfit.models.piecewise_ridge import make_piecewise_adapter
from tetris_rl.rewardfit.types import FitResult


def _build_registry() -> Dict[str, ModelAdapter]:
    reg: Dict[str, ModelAdapter] = {}

    for ad in make_linear_adapters():
        reg[str(ad.kind).lower().strip()] = ad

    pw = make_piecewise_adapter()
    reg[str(pw.kind).lower().strip()] = pw

    iso = make_isotonic_adapter()
    reg[str(iso.kind).lower().strip()] = iso

    mlp = make_mlp_adapter()
    reg[str(mlp.kind).lower().strip()] = mlp

    return reg


MODEL_REGISTRY: Dict[str, ModelAdapter] = _build_registry()


def get_adapter(kind: str) -> ModelAdapter:
    k = str(kind).lower().strip()
    if k not in MODEL_REGISTRY:
        raise ValueError(f"unknown model kind: {kind!r}")
    return MODEL_REGISTRY[k]


def get_adapter_for_fit(*, fit: FitResult, fallback_kind: str) -> ModelAdapter:
    k = str(fit.extra.get("kind", "")).lower().strip()
    if not k:
        k = str(fallback_kind).lower().strip()
    return get_adapter(k)


def fit_models(
        X: np.ndarray,
        y: np.ndarray,
        *,
        logger: Any,
        which: str,
        feature_names: Sequence[str],
        fit_intercept: bool = True,
) -> List[FitResult]:
    """
    Thin registry dispatch. Validation stays here so model files stay focused.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch: {X.shape} vs {y.shape}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 rows to fit")

    k = str(which).lower().strip()
    ad = get_adapter(k)

    # IMPORTANT: forward fit_intercept so CLI --no-bias is honored by models_old that support it.
    return ad.fit(
        X=X,
        y=y,
        logger=logger,
        which=k,
        feature_names=feature_names,
        fit_intercept=bool(fit_intercept),
    )
