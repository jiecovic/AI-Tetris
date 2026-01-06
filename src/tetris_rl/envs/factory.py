# src/tetris_rl/envs/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.config.instantiate import instantiate
from tetris_rl.envs.catalog import ENV_REGISTRY, REWARD_REGISTRY, WARMUP_REGISTRY


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
    warmup_fn: Any | None = None


def build_env(*, cfg: Dict[str, Any], env_cfg: Dict[str, Any], game: Any) -> BuiltEnv:
    """
    Build a single env instance.

    North Star:
      - Env emits RAW Dict observations only.
      - Tokenization is model-owned.
      - Warmup is a module (like reward), not env init args.
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
    # warmup (optional module)
    # ------------------------------------------------------------------
    warmup_fn = None
    if "warmup" in env_cfg and env_cfg["warmup"] is not None:
        warmup_fn = instantiate(
            spec_obj=env_cfg["warmup"],
            registry=WARMUP_REGISTRY,
            where="env.warmup",
            injected={},
        )

    # ------------------------------------------------------------------
    # env
    # ------------------------------------------------------------------
    injected_env: Dict[str, Any] = {
        "game": game,
        "reward_fn": reward_fn,
        "warmup": warmup_fn,  # MacroTetrisEnv must accept warmup: WarmupFn|None
    }

    env = instantiate(
        spec_obj=env_cfg,  # expects env.type + env.params
        registry=ENV_REGISTRY,
        where="env",
        injected=injected_env,
    )

    return BuiltEnv(env=env, reward_fn=reward_fn, warmup_fn=warmup_fn)


def make_env_from_cfg(*, cfg: Dict[str, Any], game: Any) -> BuiltEnv:
    """
    Convenience wrapper used by training / evaluation.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    env_cfg = cfg.get("env", None)
    if not isinstance(env_cfg, dict):
        raise TypeError("cfg.env must be a mapping")

    return build_env(cfg=cfg, env_cfg=env_cfg, game=game)


__all__ = ["BuiltEnv", "make_env_from_cfg"]
