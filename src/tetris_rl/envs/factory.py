# src/tetris_rl/envs/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from tetris_rl.config.instantiate import instantiate
from tetris_rl.envs.catalog import ENV_REGISTRY, REWARD_REGISTRY
from tetris_rl.game.factory import GameBundle, make_game_bundle_from_cfg


@dataclass(frozen=True)
class BuiltEnv:
    """
    Result of env factory.

    NOTE:
      - No tokenizer here (model-owned).
      - Env must emit RAW Dict observations only.
    """
    env: Any
    reward_fn: Any
    game: Any


def _env_kwargs_compat(*, env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compat shim for renamed env params.

    This runs *before* instantiate(env), so env classes can stay strict.
    """
    out: Dict[str, Any] = dict(env_cfg)

    params = out.get("params", None)
    if isinstance(params, dict):
        p = dict(params)

        # Old name in configs: illegal_action_policy
        # New MacroTetrisEnv ctor kw: invalid_action_policy
        if "illegal_action_policy" in p and "invalid_action_policy" not in p:
            p["invalid_action_policy"] = p.pop("illegal_action_policy")

        out["params"] = p

    # Back-compat: env.warmup used to exist.
    # We intentionally DO NOT pass env.warmup into env ctor; warmup is now an engine concern.
    if "warmup" in out:
        out = dict(out)
        out.pop("warmup", None)

    return out


def _maybe_migrate_env_warmup_to_game(*, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Back-compat migration:
      - if cfg.env.warmup exists AND cfg.game.warmup is missing, treat env.warmup as game.warmup.
      - remove cfg.env.warmup so env instantiation never sees it.
    """
    if not isinstance(cfg, dict):
        return cfg

    env = cfg.get("env", None)
    if not isinstance(env, dict):
        return cfg

    if "warmup" not in env:
        return cfg

    game = cfg.get("game", None)
    game_is_missing = (game is None) or (not isinstance(game, dict)) or ("warmup" not in game)

    if not game_is_missing:
        # still remove env.warmup so we don't accidentally pass it into env ctor
        cfg2 = dict(cfg)
        env2 = dict(env)
        env2.pop("warmup", None)
        cfg2["env"] = env2
        return cfg2

    cfg2 = dict(cfg)
    env2 = dict(env)
    warmup_obj = env2.pop("warmup", None)
    cfg2["env"] = env2

    game2 = dict(game) if isinstance(game, dict) else {}
    game2["warmup"] = warmup_obj
    cfg2["game"] = game2

    return cfg2


def _make_warmup_fn(bundle: GameBundle) -> Optional[Callable[..., Any]]:
    """
    Create a WarmupFn-compatible callable:
      warmup(game, rng) -> WarmupSpec | None

    - Uses bundle.warmup_prob to gate per episode.
    - Returns bundle.warmup_spec when applying warmup.
    """
    p = float(bundle.warmup_prob)
    spec = bundle.warmup_spec

    if spec is None or p <= 0.0:
        return None

    def warmup_fn(*, game: Any, rng: Any) -> Any:
        # rng is numpy Generator (gymnasium self.np_random)
        try:
            u = float(rng.random())
        except Exception:
            # ultra defensive fallback
            u = 0.0
        return spec if u < p else None

    return warmup_fn


def build_env(*, cfg: Dict[str, Any], env_cfg: Dict[str, Any], game: Any, warmup: Any) -> BuiltEnv:
    """
    Build a single env instance.

    North Star:
      - Env emits RAW Dict observations only.
      - Tokenization is model-owned.
      - Warmup noise is owned by Rust engine; Python only gates whether to apply it per episode.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")
    if not isinstance(env_cfg, dict):
        raise TypeError(f"env_cfg must be a mapping, got {type(env_cfg)!r}")
    if "reward" not in env_cfg:
        raise KeyError("env.reward missing")

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    reward_fn = instantiate(
        spec_obj=env_cfg["reward"],
        registry=REWARD_REGISTRY,
        where="env.reward",
        injected={},
    )

    # ------------------------------------------------------------------
    # env
    # ------------------------------------------------------------------
    injected_env: Dict[str, Any] = {
        "game": game,
        "reward_fn": reward_fn,
        # MacroTetrisEnv supports warmup: WarmupFn | None
        "warmup": warmup,
    }

    env_cfg2 = _env_kwargs_compat(env_cfg=env_cfg)

    env = instantiate(
        spec_obj=env_cfg2,  # expects env.type + env.params
        registry=ENV_REGISTRY,
        where="env",
        injected=injected_env,
    )

    return BuiltEnv(env=env, reward_fn=reward_fn, game=game)


def make_env_from_cfg(*, cfg: Dict[str, Any], seed: Optional[int] = None) -> BuiltEnv:
    """
    Convenience wrapper used by training / evaluation / watch.

    IMPORTANT:
      - Owns engine creation (one engine per env instance).
      - Uses cfg.game.warmup to build a warmup gate callable (probability) + WarmupSpec.
      - Use `seed` to override cfg.game.seed for this env instance (useful for VecEnv ranks).
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    cfg2 = _maybe_migrate_env_warmup_to_game(cfg=cfg)

    env_cfg = cfg2.get("env", None)
    if not isinstance(env_cfg, dict):
        raise TypeError("cfg.env must be a mapping")

    # Build a fresh engine per env instance (VecEnv-safe).
    if seed is None:
        bundle = make_game_bundle_from_cfg(cfg2)
    else:
        cfg3: Dict[str, Any] = dict(cfg2)
        game_cfg = cfg3.get("game", {}) or {}
        if not isinstance(game_cfg, dict):
            game_cfg = {}
        game_cfg2 = dict(game_cfg)
        game_cfg2["seed"] = int(seed)
        cfg3["game"] = game_cfg2
        bundle = make_game_bundle_from_cfg(cfg3)

    warmup_fn = _make_warmup_fn(bundle)

    return build_env(cfg=cfg2, env_cfg=env_cfg, game=bundle.game, warmup=warmup_fn)


__all__ = ["BuiltEnv", "build_env", "make_env_from_cfg"]
