# src/tetris_rl/training/model_io.py
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from tetris_rl.training.config import AlgoConfig


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    algo_type: str  # "ppo" | "maskable_ppo" | "dqn"
    ckpt: Path


def _pkg_version(name: str) -> str:
    try:
        return str(metadata.version(name))
    except Exception:
        return "unknown"


def _looks_like_policy_mismatch(err: Exception) -> bool:
    """
    Common when a PPO checkpoint is accidentally loaded via MaskablePPO.

    Example:
      ValueError: Policy must subclass MaskableActorCriticPolicy
    """
    if not isinstance(err, ValueError):
        return False
    s = str(err)
    return (
            "MaskableActorCriticPolicy" in s
            or "Policy must subclass" in s
            or ("maskable" in s.lower() and "policy" in s.lower())
    )


def _raise_maskable_load_hint(err: Exception) -> None:
    """
    We keep this as a single place to format errors, but we avoid blaming
    version mismatch when the signature indicates a wrong loader.
    """
    if _looks_like_policy_mismatch(err):
        msg = (
            "\n"
            "MaskablePPO load failed because the checkpoint does not use a maskable policy.\n"
            "\n"
            "This usually happens when training fell back to standard PPO at runtime\n"
            "(e.g. maskable_ppo + multidiscrete -> PPO), but your config still says\n"
            "algo.type='maskable_ppo'.\n"
            "\n"
            "Fix options:\n"
            "  - Use algo.type='ppo' for this run, OR\n"
            "  - Use env.params.action_mode='discrete' if you want masking.\n"
            "\n"
            f"Original error: {type(err).__name__}: {err}\n"
        )
        raise RuntimeError(msg) from err

    sb3_v = _pkg_version("stable-baselines3")
    contrib_v = _pkg_version("stable-baselines3-contrib")
    msg = (
        "\n"
        "MaskablePPO load failed.\n"
        "\n"
        "If you recently upgraded/downgraded SB3 packages, ensure stable-baselines3 and\n"
        "stable-baselines3-contrib are on a compatible release line.\n"
        f"  stable-baselines3:         {sb3_v}\n"
        f"  stable-baselines3-contrib: {contrib_v}\n"
        "\n"
        "Fix: install matching versions (same major/minor line), e.g.\n"
        "  pip install -U stable-baselines3 stable-baselines3-contrib\n"
        "or pin both explicitly to the same release line.\n"
        "\n"
        f"Original error: {type(err).__name__}: {err}\n"
    )
    raise RuntimeError(msg) from err


def _try_load_ppo(*, ckpt: Path, device: str) -> LoadedModel:
    from stable_baselines3 import PPO

    model = PPO.load(str(ckpt), device=str(device))
    return LoadedModel(model=model, algo_type="ppo", ckpt=ckpt)


def _try_load_dqn(*, ckpt: Path, device: str) -> LoadedModel:
    from stable_baselines3 import DQN

    model = DQN.load(str(ckpt), device=str(device))
    return LoadedModel(model=model, algo_type="dqn", ckpt=ckpt)


def load_model_from_algo_config(*, algo_cfg: AlgoConfig, ckpt: Path, device: str = "auto") -> LoadedModel:
    """
    Load a trained model checkpoint according to algo.type.

    Supported:
      - ppo
      - maskable_ppo (sb3_contrib)
      - dqn

    Notes:
      - Some training configurations may fall back to PPO at runtime even if the
        requested algo is maskable_ppo (e.g. unsupported action space wiring).
        In that case, a checkpoint saved by PPO cannot be loaded with MaskablePPO.
        We detect the common policy mismatch and fall back to PPO loading with a
        clear diagnostic.
    """
    algo_type = str(algo_cfg.type).strip().lower()
    ckpt = Path(ckpt)

    if algo_type == "ppo":
        return _try_load_ppo(ckpt=ckpt, device=str(device))

    if algo_type == "dqn":
        return _try_load_dqn(ckpt=ckpt, device=str(device))

    if algo_type == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "algo.type is 'maskable_ppo' but stable-baselines3-contrib is not installed.\n"
                "Install: pip install -U stable-baselines3-contrib"
            ) from e

        # First try the requested loader.
        try:
            model = MaskablePPO.load(str(ckpt), device=str(device))
            return LoadedModel(model=model, algo_type="maskable_ppo", ckpt=ckpt)
        except Exception as e:
            # If this is the common "wrong loader" case, fall back to PPO and return
            # the effective algo used by the checkpoint.
            if _looks_like_policy_mismatch(e):
                try:
                    loaded = _try_load_ppo(ckpt=ckpt, device=str(device))
                except Exception:
                    # PPO also failed; surface the real MaskablePPO error with guidance.
                    _raise_maskable_load_hint(e)
                    raise  # unreachable
                return loaded

            # Otherwise, raise a helpful hint (often version skew or incompatible classes).
            _raise_maskable_load_hint(e)
            raise  # unreachable

    raise ValueError(
        f"unsupported algo.type: {algo_type!r} (expected 'ppo' or 'maskable_ppo' or 'dqn')"
    )


def warn_if_maskable_with_multidiscrete(*, algo_cfg: AlgoConfig, env: Any) -> None:
    """
    Informational warning: action masking in this project is implemented for a flat
    Discrete(rot×col) action space. With MultiDiscrete you cannot enforce joint
    (rot,col) legality via a single mask.

    NOTE: This is only a warning. Loader/runtime resolution should ultimately be
    centralized at the boundary so algo, model, eval, and watch agree.
    """
    algo_type = str(algo_cfg.type).strip().lower()
    if algo_type != "maskable_ppo":
        return

    action_mode = str(getattr(env, "action_mode", "")).strip().lower()
    if action_mode == "multidiscrete":
        print(
            "[WARN] algo.type=maskable_ppo but env.action_mode=multidiscrete.\n"
            "[WARN] MaskablePPO masking is defined over a single Discrete distribution.\n"
            "[WARN] With MultiDiscrete you cannot enforce joint (rot,col) legality via masks.\n"
            "[WARN] Fix: set env.params.action_mode: discrete OR use algo.type=ppo."
        )


__all__ = ["LoadedModel", "load_model_from_algo_config", "warn_if_maskable_with_multidiscrete"]
