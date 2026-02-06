from __future__ import annotations

"""
Config merge helpers.

We use this for "eval-only overrides" in two places:
  - training/evaluation/eval_runner.py (eval env semantics)
  - cli/watch.py / cli/benchmark.py (optional: run with eval semantics)

Canonical override shape:
  env_override:
    game: {...}   # merged into cfg.game
    env:  {...}   # merged into cfg.env

Key semantic rule (component-aware):
  - If an override dict specifies {"type": ...}, that component is REPLACED (not deep-merged),
    to avoid leaking base params into disabled/swapped components.
"""

from typing import Any, Dict


# -----------------------------------------------------------------------------
# low-level merge primitives
# -----------------------------------------------------------------------------

def _merge_component(base: Any, override: Any) -> Any:
    """
    Component-aware merge:
      - If override is not a dict: overwrite.
      - If override specifies "type": REPLACE the whole component.
      - Otherwise deep-merge recursively.
    """
    if not isinstance(override, dict):
        return override
    if not isinstance(base, dict):
        base = {}

    # "type" => replace component (but keep params default)
    if "type" in override:
        out = dict(override)
        # many of our component specs are {type, params}; keep params stable if omitted
        if "params" not in out:
            out["params"] = {}
        return out

    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_component(out[k], v)
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def merge_cfg_for_eval(*, cfg: Dict[str, Any], env_override: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Return a new cfg dict patched for eval/watch semantics.

    - env_override is the object from cfg.train.eval.env_override (or similar).
    - Only canonical split override is accepted: {game:..., env:...}

    This merges into root keys:
      cfg["game"] and cfg["env"].
    """
    cfg_eval: Dict[str, Any] = dict(cfg)

    ov = env_override or {}
    if not isinstance(ov, dict) or not ov:
        return cfg_eval

    if "game" not in ov and "env" not in ov:
        raise ValueError("env_override must contain at least one of: game, env")

    game_ov = ov.get("game", {}) or {}
    env_ov = ov.get("env", {}) or {}
    if not isinstance(game_ov, dict):
        raise TypeError("env_override.game must be a mapping")
    if not isinstance(env_ov, dict):
        raise TypeError("env_override.env must be a mapping")

    base_game = cfg_eval.get("game", {}) or {}
    if not isinstance(base_game, dict):
        base_game = {}
    base_env = cfg_eval.get("env", {}) or {}
    if not isinstance(base_env, dict):
        base_env = {}

    # Component-aware merge inside each root
    if game_ov:
        cfg_eval["game"] = _merge_component(base_game, game_ov)
    if env_ov:
        cfg_eval["env"] = _merge_component(base_env, env_ov)

    return cfg_eval


__all__ = ["merge_cfg_for_eval"]
