# src/tetris_rl/rewardfit/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np

ModelKind = Literal[
    "linear",
    "ridge",
    "lasso",
    "huber",
    "piecewise_ridge",
    "isotonic_gam",
    "mlp",
    "all",
]
ProgressKind = Literal["none", "shards", "states"]
SortWeightsKind = Literal["none", "abs"]
NormalizeKind = Literal["none", "maxabs", "l1", "std"]
SplitKind = Literal["none", "state", "shard"]


@dataclass(frozen=True)
class RewardFitConfig:
    tau: float = 1.0
    topk_actions: int = 0
    max_states: int = 0
    max_rows: int = 0
    seed: int = 0
    shards: Optional[Sequence[int]] = None
    model: ModelKind = "linear"
    progress: ProgressKind = "shards"

    # model controls
    fit_intercept: bool = True

    # feature selection (names come from manifest.feature_names)
    features: Optional[Sequence[str]] = None  # allowlist; None = all
    drop_features: Optional[Sequence[str]] = None  # denylist applied after allowlist

    # evaluation split
    split: SplitKind = "state"  # default: avoid leakage by splitting per-state
    eval_frac: float = 0.10
    test_frac: float = 0.0

    # reporting controls
    print_weights: bool = True
    print_snippet: bool = True
    sort_weights: SortWeightsKind = "abs"

    # post-fit normalization (reward-shaping friendliness)
    normalize: NormalizeKind = "std"


@dataclass(frozen=True)
class FitResult:
    name: str
    coef: np.ndarray  # (D,)
    bias: float
    r2: float

    # the feature names for coef (linear uses original, piecewise uses expanded)
    feature_names: list[str] = field(default_factory=list)

    # optional model metadata (knots, mapping, etc.)
    # NOTE: Any (not object) avoids PyCharm/typing “float(object)” issues downstream.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardFitOutput:
    """
    All outputs needed by the CLI in a single struct.
    """
    best: FitResult
    fits_sorted: list[FitResult]
    feature_names: list[str]  # original dataset feature names
    X: np.ndarray
    y: np.ndarray
    states_used: int
    rows_used: int
    save_dict: dict[str, object]
