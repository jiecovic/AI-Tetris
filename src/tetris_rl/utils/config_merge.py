# src/tetris_rl/utils/config_merge.py
from __future__ import annotations

"""
Config merge helpers.

We use this for "eval-only overrides" in two places:
  - training/evaluation/eval_runner.py (eval env semantics)
  - cli/watch.py (optional: run with eval semantics)

New canonical override shape:
  env_override:
    game: {...}   # merged into cfg.game
    env:  {...}   # merged into cfg.env

Back-compat accepted:
  env_override: {...} with keys like:
    - warmup: ...
    - params/reward/type: ...   (env-like)
  This is normalized into the new split.

Key semantic rule (component-aware):
  - If an override dict specifies {"type": ...}, that component is REPLACED (not deep-merged),
    to avoid leaking base params into disabled/swapped components.
"""

from typing import Any, Dict, Tuple


# -----------------------------------------------------------------------------
# low-level merge primitives
# -----------------------------------------------------------------------------

def _deep_merge_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge dict b into dict a. Returns a new dict; does not mutate inputs.
    Dict-only: non-dicts overwrite.
    """
    out: dict[str, Any] = dict(a)
    for k, vb in b.items():
        va = out.get(k, None)
        if isinstance(va, dict) and isinstance(vb, dict):
            out[k] = _deep_merge_dict(va, vb)
        else:
            out[k] = vb
    return out


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
# override normalization (legacy -> canonical split)
# -----------------------------------------------------------------------------

def _normalize_warmup_disable(obj: Any) -> Any:
    """
    Normalize "disable warmup" idioms.

    Accept:
      warmup: null
      warmup: {type: null}
      warmup: {type: "null"}
      warmup: {enabled: false}
      warmup: {prob: 0.0, ...}

    Return:
      None if disabled, else original object.
    """
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return obj

    # type: null / "null" / "none"
    t = obj.get("type", None)
    if t is None:
        # might still be disabled via enabled/prob
        pass
    else:
        ts = str(t).strip().lower()
        if ts in {"null", "none", "off", "disabled"}:
            return None

    if "enabled" in obj and not bool(obj.get("enabled", True)):
        return None
    if "prob" in obj:
        try:
            if float(obj.get("prob", 1.0)) <= 0.0:
                return None
        except Exception:
            # ignore weird values; let downstream validation handle it if needed
            pass

    return obj


def _split_env_override(env_override: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert env_override (possibly legacy) into (game_override, env_override).

    Canonical accepted:
      {"game": {...}, "env": {...}}

    Legacy accepted examples:
      {"warmup": {...}, "params": {...}, "type": "...", "reward": {...}}
      {"params": {...}}
      {"warmup": null}
    """
    if not isinstance(env_override, dict) or not env_override:
        return {}, {}

    # New canonical split
    if "game" in env_override or "env" in env_override:
        g = env_override.get("game", {}) or {}
        e = env_override.get("env", {}) or {}
        if not isinstance(g, dict):
            raise TypeError("env_override.game must be a mapping")
        if not isinstance(e, dict):
            raise TypeError("env_override.env must be a mapping")

        # normalize warmup disable idioms
        if "warmup" in g:
            g = dict(g)
            g["warmup"] = _normalize_warmup_disable(g.get("warmup", None))
        return dict(g), dict(e)

    # Legacy: treat warmup as game override; everything else as env override
    g_ov: Dict[str, Any] = {}
    e_ov: Dict[str, Any] = {}

    for k, v in env_override.items():
        if k == "warmup":
            g_ov["warmup"] = _normalize_warmup_disable(v)
        else:
            e_ov[k] = v

    return g_ov, e_ov


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def merge_cfg_for_eval(*, cfg: Dict[str, Any], env_override: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Return a new cfg dict patched for eval/watch semantics.

    - env_override is the object from cfg.train.eval.env_override (or similar).
    - Supports canonical split override: {game:..., env:...}
    - Supports legacy override: {warmup:..., params/type/reward:...}

    This merges into root keys:
      cfg["game"] and cfg["env"].
    """
    cfg_eval: Dict[str, Any] = dict(cfg)

    ov = env_override or {}
    if not isinstance(ov, dict) or not ov:
        return cfg_eval

    game_ov, env_ov = _split_env_override(ov)

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


# Backward-compat alias: old name was merge_env_for_eval(cfg, env_override) and only merged cfg["env"].
def merge_env_for_eval(*, cfg: Dict[str, Any], env_override: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Back-compat wrapper.

    Old contract: env_override has shape of cfg.env.
    New contract: env_override can be split {game, env} or legacy.

    We now return a patched FULL cfg (root), which is what callers should want anyway.
    """
    return merge_cfg_for_eval(cfg=cfg, env_override=env_override)


__all__ = ["merge_cfg_for_eval", "merge_env_for_eval"]
