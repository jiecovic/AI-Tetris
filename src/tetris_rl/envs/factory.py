# src/tetris_rl/envs/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from tetris_rl.config.instantiate import instantiate
from tetris_rl.envs.catalog import ENV_REGISTRY, REWARD_REGISTRY
from tetris_rl.game.factory import make_game_from_cfg


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

    # Warmup is now handled by the Rust engine (game.warmup / engine.reset defaults).
    # Keep env.warmup in configs for now (ignored), so old runs don't crash.
    if "warmup" in out:
        out = dict(out)
        out.pop("warmup", None)

    return out


def build_env(*, cfg: Dict[str, Any], env_cfg: Dict[str, Any], game: Any) -> BuiltEnv:
    """
    Build a single env instance.

    North Star:
      - Env emits RAW Dict observations only.
      - Tokenization is model-owned.
      - Warmup is owned by Rust engine, not Python env.
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
      - Use `seed` to override cfg.game.seed for this env instance (useful for VecEnv ranks).
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    env_cfg = cfg.get("env", None)
    if not isinstance(env_cfg, dict):
        raise TypeError("cfg.env must be a mapping")

    # Build a fresh engine per env instance (VecEnv-safe).
    if seed is None:
        game = make_game_from_cfg(cfg)
    else:
        cfg2: Dict[str, Any] = dict(cfg)
        game_cfg = cfg2.get("game", {}) or {}
        if not isinstance(game_cfg, dict):
            game_cfg = {}
        game_cfg2 = dict(game_cfg)
        game_cfg2["seed"] = int(seed)
        cfg2["game"] = game_cfg2
        game = make_game_from_cfg(cfg2)

    return build_env(cfg=cfg, env_cfg=env_cfg, game=game)


__all__ = ["BuiltEnv", "build_env", "make_env_from_cfg"]
