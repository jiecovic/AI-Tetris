# src/tetris_rl/env_bundles/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from tetris_rl.config.instantiate import instantiate
from tetris_rl.envs.catalog import ENV_REGISTRY, REWARD_REGISTRY
from tetris_rl.envs.config import EnvConfig
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


def build_env(*, cfg: Dict[str, Any], env_cfg: EnvConfig, game: Any, warmup: Any) -> BuiltEnv:
    """
    Build a single env instance.

    Invariants:
      - Env emits RAW Dict observations only.
      - Tokenization is model-owned.
      - Warmup noise is owned by Rust engine; Python only gates whether to apply it per episode.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")
    if not isinstance(env_cfg, EnvConfig):
        raise TypeError(f"env_cfg must be an EnvConfig, got {type(env_cfg)!r}")

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    reward_fn = instantiate(
        spec_obj=env_cfg.reward,
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

    env = instantiate(
        spec_obj=env_cfg,  # expects env.type + env.params
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
      - Uses cfg.env.game.warmup to build a warmup gate callable (probability) + WarmupSpec.
      - Use `seed` to override cfg.env.game.seed for this env instance (useful for VecEnv ranks).
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    env_cfg_raw = cfg.get("env", None)
    if not isinstance(env_cfg_raw, dict):
        raise TypeError("cfg.env must be a mapping")
    env_cfg = EnvConfig.model_validate(env_cfg_raw)

    # Build a fresh engine per env instance (VecEnv-safe).
    cfg_effective = cfg
    if seed is None:
        bundle = make_game_bundle_from_cfg(cfg)
    else:
        cfg3: Dict[str, Any] = dict(cfg)
        env_cfg3 = cfg3.get("env", {}) or {}
        if not isinstance(env_cfg3, dict):
            env_cfg3 = {}
        game_cfg = env_cfg3.get("game", {}) or {}
        if not isinstance(game_cfg, dict):
            game_cfg = {}
        game_cfg2 = dict(game_cfg)
        game_cfg2["seed"] = int(seed)
        env_cfg3 = dict(env_cfg3)
        env_cfg3["game"] = game_cfg2
        cfg3["env"] = env_cfg3
        cfg_effective = cfg3
        bundle = make_game_bundle_from_cfg(cfg3)

    warmup_fn = _make_warmup_fn(bundle)

    return build_env(cfg=cfg_effective, env_cfg=env_cfg, game=bundle.game, warmup=warmup_fn)


__all__ = ["BuiltEnv", "build_env", "make_env_from_cfg"]
