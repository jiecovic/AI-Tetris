# src/tetris_rl/utils/config_merge.py
from __future__ import annotations

"""
Config merge helpers.

We use this for "eval-only env overrides" in two places:
  - training/evaluation/eval_runner.py (intermediate eval env)
  - cli/watch.py (optional: run in eval env semantics)

Key semantic rule:
  - If an override for a nested component specifies {"type": ...}, that component is
    treated as REPLACED (not deep-merged), to avoid leaking base params into
    disabled components (e.g. warmup: {type: null}).
"""

from typing import Any, Dict


def _deep_merge(a: Any, b: Any) -> Any:
    """
    Deep-merge b into a (dict-only). Non-dicts overwrite.
    Returns a new object; does not mutate inputs.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out: dict = dict(a)
    for k, vb in b.items():
        if k in out:
            out[k] = _deep_merge(out[k], vb)
        else:
            out[k] = vb
    return out


def _merge_component(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Component-aware merge:
      - If override specifies "type", REPLACE the whole component.
      - Otherwise deep-merge, recursing into sub-components.
    """
    if "type" in override:
        out = dict(override)
        out.setdefault("params", {})
        return out

    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_component(out[k], v)
        else:
            out[k] = v
    return out


def merge_env_for_eval(*, cfg: Dict[str, Any], env_override: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Return a new cfg dict where cfg["env"] is patched by env_override (eval-only).

    env_override is expected to have the SAME shape as cfg.env (i.e. keys under "env"),
    not a full config root.
    """
    cfg_eval: Dict[str, Any] = dict(cfg)

    ov = env_override or {}
    if not isinstance(ov, dict) or not ov:
        return cfg_eval

    base_env = cfg_eval.get("env", {}) or {}
    if not isinstance(base_env, dict):
        base_env = {}

    cfg_eval["env"] = _merge_component(base_env, ov)
    return cfg_eval


__all__ = ["merge_env_for_eval"]
