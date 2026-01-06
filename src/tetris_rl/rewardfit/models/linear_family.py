# src/tetris_rl/rewardfit/models_old/linear_family.py
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from tetris_rl.rewardfit.models.adapters import ModelAdapter
from tetris_rl.rewardfit.types import FitResult


def _unscale_from_pipeline(pipe: Any) -> tuple[np.ndarray, float]:
    """
    Convert a (StandardScaler -> linear model) pipeline back into a model in the
    ORIGINAL feature space:

      X_scaled = (X - mu) / sigma
      y = w_std · X_scaled + b_std

    => y = (w_std/sigma) · X  + (b_std - (w_std/sigma)·mu)

    Works for StandardScaler(with_mean True/False).
    """
    scaler = pipe.named_steps["standardscaler"]
    model_key = next(k for k in pipe.named_steps.keys() if k != "standardscaler")
    model = pipe.named_steps[model_key]

    w_std = np.asarray(getattr(model, "coef_"), dtype=np.float64).reshape(-1)

    # sklearn uses intercept_=0.0 when fit_intercept=False, but be defensive
    b_std = float(getattr(model, "intercept_", 0.0) or 0.0)

    sigma = np.asarray(getattr(scaler, "scale_", np.ones_like(w_std)), dtype=np.float64).reshape(-1)

    mu_attr = getattr(scaler, "mean_", None)
    if mu_attr is None:
        mu = np.zeros_like(w_std, dtype=np.float64)
    else:
        mu = np.asarray(mu_attr, dtype=np.float64).reshape(-1)

    safe = sigma > 0
    w = np.zeros_like(w_std, dtype=np.float64)
    w[safe] = w_std[safe] / sigma[safe]

    b = b_std - float(np.sum((w_std[safe] * mu[safe]) / sigma[safe]))
    return w, float(b)


def _fit_one_linear(
        *,
        X: np.ndarray,
        y: np.ndarray,
        logger: Any,
        name: str,
        kind: str,
        base_model: Any,
        feature_names: Sequence[str],
        fit_intercept: bool,
) -> FitResult:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # If we do NOT fit an intercept in ORIGINAL feature space, we also must NOT mean-center X.
    with_mean = bool(fit_intercept)

    t0 = time.perf_counter()
    pipe = make_pipeline(StandardScaler(with_mean=with_mean, with_std=True), base_model)
    pipe.fit(X, y)
    dt = time.perf_counter() - t0

    r2 = float(pipe.score(X, y))
    coef, bias = _unscale_from_pipeline(pipe)

    # Enforce contract: "no-bias" => strictly through origin in ORIGINAL space.
    if not bool(fit_intercept):
        bias = 0.0

    logger.info("[fit_reward] done %s in %0.2fs (train R²=%0.4f)", name, dt, r2)

    return FitResult(
        name=name + " + StandardScaler",
        coef=np.asarray(coef, dtype=np.float64),
        bias=float(bias),
        r2=float(r2),
        feature_names=list(feature_names),
        extra={"kind": str(kind), "fit_intercept": bool(fit_intercept)},
    )


def _fit_linear_family_impl(
        X: np.ndarray,
        y: np.ndarray,
        *,
        logger: Any,
        which: str,
        feature_names: Sequence[str],
        fit_intercept: bool,
) -> list[FitResult]:
    try:
        from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from e

    fi = bool(fit_intercept)

    specs: dict[str, tuple[str, Any]] = {
        "linear": ("LinearRegression", LinearRegression(fit_intercept=fi)),
        "ridge": ("Ridge(alpha=1.0)", Ridge(alpha=1.0, fit_intercept=fi)),
        "lasso": ("Lasso(alpha=1e-3)", Lasso(alpha=1e-3, fit_intercept=fi, max_iter=20000)),
        "huber": ("HuberRegressor", HuberRegressor(fit_intercept=fi)),
    }

    w = str(which).lower().strip()
    if w == "all":
        keys = ["linear", "ridge", "lasso", "huber"]
    else:
        if w not in specs:
            raise ValueError(f"unknown model kind: {which!r}")
        keys = [w]

    out: list[FitResult] = []
    for i, k in enumerate(keys, start=1):
        name, model = specs[k]
        if len(keys) == 1:
            logger.info("[fit_reward] fitting %s", name)
        else:
            logger.info("[fit_reward] fitting %s (%d/%d)", name, i, len(keys))
        out.append(
            _fit_one_linear(
                X=X,
                y=y,
                logger=logger,
                name=name,
                kind=k,
                base_model=model,
                feature_names=feature_names,
                fit_intercept=fi,
            )
        )
    return out


def _fit_linear_family(
        X: np.ndarray,
        y: np.ndarray,
        *,
        logger: Any,
        which: str,
        feature_names: Sequence[str],
        fit_intercept: bool = True,
) -> list[FitResult]:
    return _fit_linear_family_impl(
        X,
        y,
        logger=logger,
        which=str(which),
        feature_names=feature_names,
        fit_intercept=bool(fit_intercept),
    )


def make_linear_adapters() -> list[ModelAdapter]:
    return [
        ModelAdapter(kind="linear", fit=_fit_linear_family),
        ModelAdapter(kind="ridge", fit=_fit_linear_family),
        ModelAdapter(kind="lasso", fit=_fit_linear_family),
        ModelAdapter(kind="huber", fit=_fit_linear_family),
        ModelAdapter(kind="all", fit=_fit_linear_family),
    ]
