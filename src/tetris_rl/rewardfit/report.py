# src/tetris_rl/rewardfit/report.py
from __future__ import annotations

from typing import Any, Sequence

from tetris_rl.rewardfit.types import FitResult


def log_fit(
    *,
    logger: Any,
    feature_names: Sequence[str],
    fit: FitResult,
    print_weights: bool,
    print_snippet: bool,
    sort_weights: str,
) -> None:
    logger.info("")
    logger.info("[fit_reward] result: %s | train_R2=%0.4f", fit.name, float(fit.r2))

    if print_weights:
        pairs = [(str(nm), float(w)) for nm, w in zip(feature_names, fit.coef)]
        if str(sort_weights).lower().strip() == "abs":
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        logger.info("[fit_reward] weights:")
        for nm, w in pairs:
            logger.info("  %-22s %+0.6f", nm, w)
        logger.info("  %-22s %+0.6f", "bias", float(fit.bias))

    if print_snippet:
        args = ", ".join(str(n) for n in feature_names)
        logger.info("")
        logger.info("[fit_reward] snippet:")
        logger.info("def learned_delta_reward(*, %s):", args)
        logger.info("    return (")
        for nm, w in zip(feature_names, fit.coef):
            logger.info("        %+0.6f * float(%s)", float(w), str(nm))
        logger.info("        %+0.6f", float(fit.bias))
        logger.info("    )")
