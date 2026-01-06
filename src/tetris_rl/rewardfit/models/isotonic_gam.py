# src/tetris_rl/rewardfit/models_old/isotonic_gam.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from tetris_rl.rewardfit.models.adapters import ModelAdapter
from tetris_rl.rewardfit.types import FitResult

ISOTONIC_GAM_KIND = "isotonic_gam"


@dataclass(frozen=True)
class IsotonicGAMSpec:
    """
    Additive monotone model via backfitting with per-feature isotonic regressors.

      r(x) = b + sum_i f_i(x_i)

    Each f_i is fit to the residual:
      resid_i = y - b - sum_{j!=i} f_j(x_j)

    Runtime predictor is cheap: sum of 1D interpolants.
    """
    num_iters: int = 6
    center_each_iter: bool = True  # keep components mean-centered for identifiability


def _interp_1d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Vectorized linear interpolation with edge clamping.
    NOTE: isotonic is piecewise-constant, but linear interp is fine as a smooth-ish shaping proxy.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    xp = np.asarray(xp, dtype=np.float64).reshape(-1)
    fp = np.asarray(fp, dtype=np.float64).reshape(-1)
    if xp.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    if xp.size == 1:
        return np.full_like(x, float(fp[0]), dtype=np.float64)
    return np.interp(x, xp, fp, left=float(fp[0]), right=float(fp[-1]))


def _predict_isotonic_gam(
    X: np.ndarray,
    *,
    bias: float,
    knots_per_feature: Sequence[np.ndarray],
    values_per_feature: Sequence[np.ndarray],
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, f = X.shape
    r = np.full((n,), float(bias), dtype=np.float64)
    for i in range(f):
        r += _interp_1d(X[:, i], knots_per_feature[i], values_per_feature[i])
    return r


def fit_isotonic_gam(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    logger: Any,
    spec: IsotonicGAMSpec,
    directions: Sequence[int],
) -> FitResult:
    try:
        from sklearn.isotonic import IsotonicRegression
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

    n, f = X.shape
    if len(feature_names) != f:
        raise ValueError(f"feature_names mismatch: {len(feature_names)} vs F={f}")
    if len(directions) != f:
        raise ValueError(f"directions mismatch: {len(directions)} vs F={f}")

    dirs = [1 if int(d) >= 0 else -1 for d in directions]

    t0 = time.perf_counter()

    bias = float(np.mean(y))
    fx = np.zeros((n, f), dtype=np.float64)

    knots_per_feature: List[np.ndarray] = [np.zeros((0,), dtype=np.float64) for _ in range(f)]
    values_per_feature: List[np.ndarray] = [np.zeros((0,), dtype=np.float64) for _ in range(f)]

    for _it in range(int(spec.num_iters)):
        for i in range(f):
            resid = y - bias - np.sum(fx, axis=1) + fx[:, i]

            xi = X[:, i]
            inc = bool(dirs[i] > 0)

            iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
            iso.fit(xi, resid)

            xp = np.asarray(getattr(iso, "X_thresholds_", None), dtype=np.float64).reshape(-1)
            fp = np.asarray(getattr(iso, "y_thresholds_", None), dtype=np.float64).reshape(-1)

            if xp.size == 0:
                lo = float(np.min(xi))
                hi = float(np.max(xi))
                m = float(np.mean(resid))
                xp = np.asarray([lo, hi], dtype=np.float64)
                fp = np.asarray([m, m], dtype=np.float64)

            fi = _interp_1d(xi, xp, fp)

            if bool(spec.center_each_iter):
                m = float(np.mean(fi))
                fi = fi - m
                fp = fp - m
                bias += m

            fx[:, i] = fi
            knots_per_feature[i] = xp
            values_per_feature[i] = fp

        if bool(spec.center_each_iter):
            bias = float(np.mean(y - np.sum(fx, axis=1)))

    dt = time.perf_counter() - t0

    yhat = float(bias) + np.sum(fx, axis=1)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 0.0 if ss_tot <= 0 else (1.0 - ss_res / ss_tot)

    logger.info(
        "[fit_reward] done IsotonicGAM(iters=%d) in %0.2fs (train R²=%0.4f)",
        int(spec.num_iters),
        dt,
        float(r2),
    )

    extra: Dict[str, object] = {
        "kind": ISOTONIC_GAM_KIND,
        "isotonic_num_iters": int(spec.num_iters),
        "directions": [int(d) for d in dirs],
        "knots_per_feature": [np.asarray(k, dtype=np.float64) for k in knots_per_feature],
        "values_per_feature": [np.asarray(v, dtype=np.float64) for v in values_per_feature],
        "predictor": "isotonic_gam_interp",
    }

    return FitResult(
        name=f"IsotonicGAM(iters={int(spec.num_iters)})",
        coef=np.zeros((0,), dtype=np.float64),
        bias=float(bias),
        r2=float(r2),
        feature_names=list(feature_names),
        extra=extra,
    )


def isotonic_gam_predict_from_fit(X: np.ndarray, fit: FitResult, base_feature_dim: int) -> np.ndarray:
    if not fit.extra or str(fit.extra.get("kind", "")).lower().strip() != ISOTONIC_GAM_KIND:
        raise ValueError(f"fit is not {ISOTONIC_GAM_KIND!r}")

    knots = fit.extra["knots_per_feature"]
    vals = fit.extra["values_per_feature"]

    if not isinstance(knots, list) or len(knots) != int(base_feature_dim):
        raise ValueError("invalid knots_per_feature in fit.extra")
    if not isinstance(vals, list) or len(vals) != int(base_feature_dim):
        raise ValueError("invalid values_per_feature in fit.extra")

    return _predict_isotonic_gam(
        np.asarray(X, dtype=np.float64),
        bias=float(fit.bias),
        knots_per_feature=[np.asarray(k, dtype=np.float64) for k in knots],
        values_per_feature=[np.asarray(v, dtype=np.float64) for v in vals],
    )


def isotonic_gam_scale_extra(extra: Dict[str, object], scale: float) -> Dict[str, object]:
    """
    Keep compact params consistent with scaled bias.
    Components are in 'values_per_feature', so they must be scaled too.
    """
    out = dict(extra)
    if str(out.get("kind", "")).lower().strip() != ISOTONIC_GAM_KIND:
        return out

    if "values_per_feature" in out:
        out["values_per_feature"] = [
            np.asarray(v, dtype=np.float64) * float(scale) for v in out["values_per_feature"]
        ]
    return out


def isotonic_gam_save_extra_npz(save_dict: Dict[str, object], best_raw: FitResult, best_scaled: FitResult) -> None:
    """
    Model-specific np.savez payload for isotonic GAM.
    """
    save_dict["best_extra_kind"] = np.asarray([str(best_raw.extra.get("kind", ""))], dtype=object)

    save_dict["isotonic_directions"] = np.asarray(best_raw.extra["directions"], dtype=np.int64)
    save_dict["isotonic_knots_per_feature"] = np.asarray(best_raw.extra["knots_per_feature"], dtype=object)
    save_dict["isotonic_values_per_feature_raw"] = np.asarray(best_raw.extra["values_per_feature"], dtype=object)
    save_dict["isotonic_values_per_feature_scaled"] = np.asarray(best_scaled.extra["values_per_feature"], dtype=object)


def _fit_isotonic_dispatch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    logger: Any,
    which: str,
    feature_names: Sequence[str],
) -> list[FitResult]:
    _ = which
    logger.info("[fit_reward] fitting IsotonicGAM")
    # Default monotone directions for your delta features:
    # lines: more is better (+)
    # holes/heights/bumpiness: more is worse (-)
    dirs = []
    for nm in feature_names:
        s = str(nm).lower()
        if "line" in s:
            dirs.append(+1)
        else:
            dirs.append(-1)

    fit = fit_isotonic_gam(
        X=X,
        y=y,
        feature_names=list(feature_names),
        logger=logger,
        spec=IsotonicGAMSpec(num_iters=6, center_each_iter=True),
        directions=dirs,
    )
    return [fit]


def make_isotonic_adapter() -> ModelAdapter:
    return ModelAdapter(
        kind=ISOTONIC_GAM_KIND,
        fit=_fit_isotonic_dispatch,
        predict=isotonic_gam_predict_from_fit,
        scale_extra=isotonic_gam_scale_extra,
        save_extra=isotonic_gam_save_extra_npz,
    )
