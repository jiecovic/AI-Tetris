# src/tetris_rl/rewardfit/models_old/piecewise_ridge.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tetris_rl.rewardfit.models.adapters import ModelAdapter
from tetris_rl.rewardfit.types import FitResult

PIECEWISE_RIDGE_KIND = "piecewise_ridge"


@dataclass(frozen=True)
class PiecewiseSpec:
    """
    Piecewise-linear per-feature model:

      r(x) = b
             + sum_i w_lin[i] * x_i
             + sum_i sum_k w_hinge[i,k] * max(0, x_i - knot[i,k])

    We fit this by expanding X into [x_i, max(0, x_i - knot[i,k])...] and using Ridge.
    """
    num_knots: int = 8
    quantile_lo: float = 0.05
    quantile_hi: float = 0.95


def _quantile_knots(x: np.ndarray, *, num_knots: int, q_lo: float, q_hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if num_knots <= 0:
        return np.zeros((0,), dtype=np.float64)

    q_lo = float(q_lo)
    q_hi = float(q_hi)
    q_lo = max(0.0, min(1.0, q_lo))
    q_hi = max(0.0, min(1.0, q_hi))
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo

    qs = np.linspace(q_lo, q_hi, num_knots + 2, dtype=np.float64)[1:-1]  # drop endpoints
    knots = np.quantile(x, qs, method="linear")
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)

    # De-duplicate (quantiles can collide if feature is discrete)
    if knots.size:
        knots = np.unique(knots)
    return knots


def _expand_piecewise(
        X: np.ndarray,
        *,
        knots_per_feature: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Build design matrix:
      [X, hinge_0_0, hinge_0_1, ..., hinge_{F-1,K-1}]
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n, f = X.shape
    if len(knots_per_feature) != f:
        raise ValueError(f"knots_per_feature length mismatch: {len(knots_per_feature)} vs F={f}")

    parts: List[np.ndarray] = [X]
    for i in range(f):
        ks = np.asarray(knots_per_feature[i], dtype=np.float64).reshape(-1)
        if ks.size == 0:
            continue
        xi = X[:, i: i + 1]  # (N,1)
        hinges = np.maximum(0.0, xi - ks.reshape(1, -1))  # (N,K)
        parts.append(hinges)

    return np.concatenate(parts, axis=1)


def _expanded_feature_names(
        base: Sequence[str],
        *,
        knots_per_feature: Sequence[np.ndarray],
) -> List[str]:
    out = [str(nm) for nm in base]
    for i, nm in enumerate(base):
        ks = np.asarray(knots_per_feature[i], dtype=np.float64).reshape(-1)
        for k, t in enumerate(ks.tolist()):
            out.append(f"{nm}__hinge_{k}@{t:.6g}")
    return out


def _unpack_piecewise_weights(
        w: np.ndarray,
        *,
        feature_dim: int,
        knots_per_feature: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Split coef vector into:
      w_lin: (F,)
      w_hinges: list length F, each (K_i,)
    where coef layout matches _expand_piecewise / _expanded_feature_names.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size < feature_dim:
        raise ValueError("coef too short for piecewise unpack")

    w_lin = w[:feature_dim].astype(np.float64, copy=False)
    offs = feature_dim
    w_hinges: List[np.ndarray] = []
    for i in range(feature_dim):
        ks = np.asarray(knots_per_feature[i], dtype=np.float64).reshape(-1)
        ki = int(ks.size)
        if ki == 0:
            w_hinges.append(np.zeros((0,), dtype=np.float64))
            continue
        wi = w[offs: offs + ki].astype(np.float64, copy=False)
        w_hinges.append(wi)
        offs += ki
    return w_lin, w_hinges


def _predict_piecewise(
        X: np.ndarray,
        *,
        bias: float,
        w_lin: np.ndarray,
        w_hinges: Sequence[np.ndarray],
        knots_per_feature: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Vectorized r(x) for normalization/std, without constructing expanded matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    n, f = X.shape
    r = np.full((n,), float(bias), dtype=np.float64)
    r += X @ np.asarray(w_lin, dtype=np.float64).reshape(-1)

    for i in range(f):
        ks = np.asarray(knots_per_feature[i], dtype=np.float64).reshape(-1)
        ws = np.asarray(w_hinges[i], dtype=np.float64).reshape(-1)
        if ks.size == 0:
            continue
        xi = X[:, i: i + 1]  # (N,1)
        hinges = np.maximum(0.0, xi - ks.reshape(1, -1))  # (N,K)
        r += hinges @ ws
    return r


def fit_piecewise_ridge(
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
        logger: Any,
        spec: PiecewiseSpec,
        alpha: float = 1.0,
) -> FitResult:
    """
    Fits piecewise ridge and returns a FitResult where:
      - coef is the expanded coefficient vector (linear terms first, then hinges)
      - fit.feature_names is the expanded feature-name list
      - fit.extra stores knots and a compact representation for inference
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from e

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch: {X.shape} vs {y.shape}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 rows to fit")

    f = int(X.shape[1])
    if len(feature_names) != f:
        raise ValueError(f"feature_names mismatch: {len(feature_names)} vs F={f}")

    knots_per_feature: List[np.ndarray] = []
    for i in range(f):
        knots_per_feature.append(
            _quantile_knots(
                X[:, i],
                num_knots=int(spec.num_knots),
                q_lo=float(spec.quantile_lo),
                q_hi=float(spec.quantile_hi),
            )
        )

    Xpw = _expand_piecewise(X, knots_per_feature=knots_per_feature)
    names_pw = _expanded_feature_names(feature_names, knots_per_feature=knots_per_feature)

    t0 = time.perf_counter()
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Ridge(alpha=float(alpha), fit_intercept=True),
    )
    pipe.fit(Xpw, y)
    dt = time.perf_counter() - t0
    r2 = float(pipe.score(Xpw, y))

    # Unscale back to original (unstandardized) Xpw coordinates.
    scaler: Any = pipe.named_steps["standardscaler"]
    model: Any = pipe.named_steps["ridge"]

    w_std = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    b_std = float(model.intercept_)

    sigma = np.asarray(getattr(scaler, "scale_", None), dtype=np.float64).reshape(-1)
    mu = np.asarray(getattr(scaler, "mean_", None), dtype=np.float64).reshape(-1)
    safe = sigma > 0

    w = np.zeros_like(w_std, dtype=np.float64)
    w[safe] = w_std[safe] / sigma[safe]
    b = b_std - float(np.sum((w_std[safe] * mu[safe]) / sigma[safe]))

    w_lin, w_hinges = _unpack_piecewise_weights(w, feature_dim=f, knots_per_feature=knots_per_feature)

    logger.info(
        "[fit_reward] done PiecewiseRidge(knots=%d,alpha=%s) in %0.2fs (train R²=%0.4f)",
        int(spec.num_knots),
        str(alpha),
        dt,
        r2,
    )

    extra: Dict[str, object] = {
        "kind": PIECEWISE_RIDGE_KIND,
        "piecewise_num_knots": int(spec.num_knots),
        "piecewise_quantile_lo": float(spec.quantile_lo),
        "piecewise_quantile_hi": float(spec.quantile_hi),
        "knots_per_feature": [np.asarray(k, dtype=np.float64) for k in knots_per_feature],
        "w_lin": np.asarray(w_lin, dtype=np.float64),
        "w_hinges": [np.asarray(wi, dtype=np.float64) for wi in w_hinges],
        "predictor": "piecewise_linear_hinge",
    }

    return FitResult(
        name=f"PiecewiseRidge(knots={int(spec.num_knots)},alpha={float(alpha)}) + StandardScaler",
        coef=np.asarray(w, dtype=np.float64),
        bias=float(b),
        r2=float(r2),
        feature_names=list(names_pw),
        extra=extra,
    )


def piecewise_predict_from_fit(X: np.ndarray, fit: FitResult, base_feature_dim: int) -> np.ndarray:
    """
    Adapter predict hook: compute r(X) for piecewise fit without expanded matrix.
    """
    if not fit.extra or str(fit.extra.get("kind", "")).lower().strip() != PIECEWISE_RIDGE_KIND:
        raise ValueError(f"fit is not {PIECEWISE_RIDGE_KIND!r}")

    knots_per_feature = fit.extra["knots_per_feature"]
    w_lin = fit.extra["w_lin"]
    w_hinges = fit.extra["w_hinges"]

    if not isinstance(knots_per_feature, list) or len(knots_per_feature) != int(base_feature_dim):
        raise ValueError("invalid knots_per_feature in fit.extra")

    return _predict_piecewise(
        np.asarray(X, dtype=np.float64),
        bias=float(fit.bias),
        w_lin=np.asarray(w_lin, dtype=np.float64),
        w_hinges=[np.asarray(wi, dtype=np.float64) for wi in w_hinges],
        knots_per_feature=[np.asarray(k, dtype=np.float64) for k in knots_per_feature],
    )


def piecewise_scale_extra(extra: Dict[str, object], scale: float) -> Dict[str, object]:
    """
    Keep compact params consistent with scaled coef/bias for inference/export.
    """
    out = dict(extra)
    if str(out.get("kind", "")).lower().strip() != PIECEWISE_RIDGE_KIND:
        return out

    if "w_lin" in out:
        out["w_lin"] = np.asarray(out["w_lin"], dtype=np.float64) * float(scale)
    if "w_hinges" in out:
        out["w_hinges"] = [np.asarray(wi, dtype=np.float64) * float(scale) for wi in out["w_hinges"]]
    return out


def piecewise_save_extra_npz(save_dict: Dict[str, object], best_raw: FitResult, best_scaled: FitResult) -> None:
    """
    Model-specific np.savez payload for piecewise ridge.
    """
    save_dict["best_extra_kind"] = np.asarray([str(best_raw.extra.get("kind", ""))], dtype=object)
    save_dict["piecewise_knots_per_feature"] = np.asarray(best_raw.extra["knots_per_feature"], dtype=object)
    save_dict["piecewise_w_lin_raw"] = np.asarray(best_raw.extra["w_lin"], dtype=np.float64)
    save_dict["piecewise_w_hinges_raw"] = np.asarray(best_raw.extra["w_hinges"], dtype=object)

    save_dict["piecewise_w_lin_scaled"] = np.asarray(best_scaled.extra["w_lin"], dtype=np.float64)
    save_dict["piecewise_w_hinges_scaled"] = np.asarray(best_scaled.extra["w_hinges"], dtype=object)


def _fit_piecewise_dispatch(
        X: np.ndarray,
        y: np.ndarray,
        *,
        logger: Any,
        which: str,
        feature_names: Sequence[str],
) -> list[FitResult]:
    _ = which  # dispatch key; single model
    logger.info("[fit_reward] fitting PiecewiseRidge")
    fit = fit_piecewise_ridge(
        X=X,
        y=y,
        feature_names=list(feature_names),
        logger=logger,
        spec=PiecewiseSpec(num_knots=8, quantile_lo=0.05, quantile_hi=0.95),
        alpha=1.0,
    )
    return [fit]


def make_piecewise_adapter() -> ModelAdapter:
    return ModelAdapter(
        kind=PIECEWISE_RIDGE_KIND,
        fit=_fit_piecewise_dispatch,
        predict=piecewise_predict_from_fit,
        scale_extra=piecewise_scale_extra,
        save_extra=piecewise_save_extra_npz,
    )
