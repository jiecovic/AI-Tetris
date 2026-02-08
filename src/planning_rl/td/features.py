# src/planning_rl/td/features.py
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def extract_features(
    *,
    env: Any,
    features: Sequence[str],
    action: Any | None = None,
    action_id: int | None = None,
    after_clear: bool = True,
    pre_clear: bool | None = None,
    visible: bool = False,
) -> np.ndarray:
    if pre_clear is not None:
        after_clear = not bool(pre_clear)
    if action is not None:
        if hasattr(env, "value_features_for_action"):
            vals = env.value_features_for_action(
                features=features,
                action=action,
                after_clear=after_clear,
                visible=visible,
            )
            return np.asarray(vals, dtype=np.float32)
        if hasattr(env, "value_features_from_action"):
            vals = env.value_features_from_action(
                features=features,
                action=action,
                after_clear=after_clear,
                visible=visible,
            )
            return np.asarray(vals, dtype=np.float32)
        raise RuntimeError("env must expose value_features_for_action for action-based features")

    if hasattr(env, "value_features"):
        vals = env.value_features(
            features=features,
            action_id=action_id,
            after_clear=after_clear,
            visible=visible,
        )
        return np.asarray(vals, dtype=np.float32)

    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env must expose .game or .value_features for TD feature extraction")

    if action_id is None:
        vals = game.heuristic_features(list(features), False, bool(visible))
        return np.asarray(vals, dtype=np.float32)

    vals = game.simulate_active_features(int(action_id), list(features), bool(after_clear), False, bool(visible))
    if vals is None:
        raise RuntimeError("simulate_active_features returned None for a requested action_id")
    return np.asarray(vals, dtype=np.float32)


__all__ = ["extract_features"]
