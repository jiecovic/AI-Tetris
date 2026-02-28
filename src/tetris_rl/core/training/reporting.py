# src/tetris_rl/core/training/reporting.py
from __future__ import annotations

import inspect
import platform
import pprint
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from stable_baselines3 import PPO

from tetris_rl.core.utils.model_params import format_sb3_param_report, format_sb3_param_summary


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.6g}"
    return f"{x:.3e}"


def _fmt_scalar(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return _fmt_int(x)
    if isinstance(x, float):
        return _fmt_float(x)
    return str(x)


def _try_relpath(value: Any, *, base: Path) -> str:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(base))
        except Exception:
            return str(value)
    if isinstance(value, str):
        try:
            p = Path(value)
            return str(p.relative_to(base))
        except Exception:
            return value
    return str(value)


def _log_block(logger, header: str, obj: Any) -> None:
    """
    Snake-style: header line, then repr(obj) line-by-line.
    IMPORTANT: caller must NOT pass the full policy object here (it duplicates everything).
    """
    logger.info(header)
    if obj is None:
        logger.info("  <None>")
        return
    for line in repr(obj).splitlines():
        logger.info(line)


def _filter_simple_params(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    data = {}
    raw = obj if isinstance(obj, dict) else getattr(obj, "__dict__", None)
    if not isinstance(raw, dict):
        return data
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            data[str(key)] = value
    return dict(sorted(data.items(), key=lambda kv: kv[0]))


def _safe_call(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    try:
        value = getattr(obj, attr)
    except Exception:
        return None
    if callable(value):
        try:
            return value()
        except Exception:
            return None
    return value


def _format_warmup_cfg(warmup_cfg: Any) -> str:
    if warmup_cfg is None:
        return "off"
    if not isinstance(warmup_cfg, dict):
        return str(warmup_cfg)
    prob = warmup_cfg.get("prob", 1.0)
    spec = warmup_cfg.get("spec", None)
    if not isinstance(spec, dict):
        return f"{spec} (prob={_fmt_scalar(prob)})"
    spec_type = spec.get("type", None)
    params = spec.get("params", {})
    if isinstance(params, dict):
        params = _filter_simple_params(params)
        params_str = ",".join(f"{k}={_fmt_scalar(v)}" for k, v in params.items())
        return f"{spec_type}(prob={_fmt_scalar(prob)}, {params_str})"
    return f"{spec_type}(prob={_fmt_scalar(prob)})"


def log_env_reward_summary(
    *,
    logger,
    label: str,
    built_env: Any,
    env_cfg: Optional[dict[str, Any]] = None,
    time_limit_steps: int | None = None,
) -> None:
    env = getattr(built_env, "env", None)
    game = getattr(built_env, "game", None)
    reward_fn = getattr(built_env, "reward_fn", None)

    action_mode = getattr(env, "action_mode", None) if env is not None else None
    max_steps = getattr(env, "max_steps", None) if env is not None else None
    invalid_policy = getattr(env, "invalid_action_policy", None) if env is not None else None
    feature_mode = getattr(env, "feature_clear_mode", None) if env is not None else None
    piece_rule = _safe_call(game, "piece_rule")

    warmup_cfg = None
    if isinstance(env_cfg, dict):
        warmup_cfg = (env_cfg.get("game", {}) or {}).get("warmup", None)
        if piece_rule is None:
            piece_rule = (env_cfg.get("game", {}) or {}).get("piece_rule", None)

    logger.info(
        "[env/%s] type=%s action_mode=%s invalid_action_policy=%s feature_clear=%s max_steps=%s time_limit=%s piece_rule=%s warmup=%s",
        str(label),
        type(env).__name__ if env is not None else "<none>",
        _fmt_scalar(action_mode),
        _fmt_scalar(invalid_policy),
        _fmt_scalar(feature_mode),
        _fmt_scalar(max_steps),
        _fmt_scalar(time_limit_steps),
        _fmt_scalar(piece_rule),
        _format_warmup_cfg(warmup_cfg),
    )

    reward_params = _filter_simple_params(reward_fn)
    params_str = ",".join(f"{k}={_fmt_scalar(v)}" for k, v in reward_params.items())
    logger.info(
        "[reward/%s] type=%s params=%s",
        str(label),
        type(reward_fn).__name__ if reward_fn is not None else "<none>",
        params_str or "-",
    )


def log_runtime_info(*, logger) -> None:
    """
    Minimal runtime info (snake-style).
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"[train] python={py} platform={platform.system()}-{platform.release()}")

    cuda = torch.cuda.is_available()
    gpu = None
    if cuda:
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "<unknown>"
    logger.info(f"[train] torch={torch.__version__} cuda={cuda} gpu={gpu}")


def _effective_ppo_init_kwargs(model: PPO) -> dict[str, Any]:
    """
    Same trick as snake: best-effort dump of PPO fields that correspond to PPO.__init__ kwargs.
    """
    try:
        sig = inspect.signature(model.__class__.__init__)
    except Exception:
        sig = inspect.signature(PPO.__init__)
    keys = [k for k in sig.parameters.keys() if k != "self"]

    out: dict[str, Any] = {}
    for k in keys:
        if k in {"policy", "env"}:
            continue
        out[k] = getattr(model, k, "<not_exposed>")
    return out


def log_ppo_params(*, model: Any, logger, tb_log: Optional[Path] = None) -> None:
    """
    Log SB3 PPO effective params in a readable way (snake-style).
    Keeps policy_kwargs readable (multi-line pretty print), but not insane-width.
    """
    repo = Path.cwd()

    logger.info("PPO effective params (SB3):")
    # Works for stable_baselines3.PPO and sb3_contrib.MaskablePPO (same init kwargs).
    eff = _effective_ppo_init_kwargs(model)

    # Force tensorboard_log to be shown nicely if we have it
    if tb_log is not None:
        eff["tensorboard_log"] = tb_log

    for k in sorted(eff.keys()):
        v = eff[k]

        if k == "policy_kwargs":
            logger.info("  policy_kwargs:")
            # Pretty format WITHOUT exploding line width
            pps = pprint.pformat(v, width=100, compact=False, sort_dicts=False)
            for line in pps.splitlines():
                logger.info(f"    {line}")
            continue

        if isinstance(v, float):
            vs = _fmt_float(v)
        elif isinstance(v, int):
            vs = _fmt_int(v)
        elif isinstance(v, (str, Path)) and (("log" in k) or ("path" in k) or k.endswith("_dir")):
            vs = _try_relpath(v, base=repo)
        else:
            vs = str(v)

        logger.info(f"  {k}: {vs}")


def log_policy_compact(*, model: Any, logger) -> None:
    """
    Compact, stable summary. No trees, no duplication.

    Supports:
      - PPO / MaskablePPO (ActorCriticPolicy-like)
    """
    policy = getattr(model, "policy", None)
    logger.info("============== Policy (compact) ==============")
    if policy is None:
        logger.info("  <no model.policy>")
        return

    def _count(m: Any) -> int:
        if not isinstance(m, torch.nn.Module):
            return 0
        return sum(int(p.numel()) for p in m.parameters() if p is not None)

    def _log_module(name: str, m: Any) -> None:
        if m is None:
            return
        logger.info(f"  {name}: {type(m).__name__} params={_fmt_int(_count(m))}")

    logger.info(f"  policy_class: {policy.__class__.__name__}")

    # ActorCriticPolicy branch (PPO / MaskablePPO)
    feat = getattr(policy, "features_extractor", None)
    pi = getattr(policy, "pi_features_extractor", None)
    vf = getattr(policy, "vf_features_extractor", None)
    mlp = getattr(policy, "mlp_extractor", None)
    an = getattr(policy, "action_net", None)
    vn = getattr(policy, "value_net", None)

    _log_module("features_extractor", feat)

    if pi is not None:
        shared = pi is feat
        msg = f"  pi_features_extractor: {type(pi).__name__} params={_fmt_int(_count(pi))}"
        if shared:
            msg += " shared_with_base=True"
        logger.info(msg)

    if vf is not None:
        shared = vf is feat
        msg = f"  vf_features_extractor: {type(vf).__name__} params={_fmt_int(_count(vf))}"
        if shared:
            msg += " shared_with_base=True"
        logger.info(msg)

    _log_module("mlp_extractor", mlp)
    _log_module("action_net", an)
    _log_module("value_net", vn)


def log_policy_full(*, model: Any, logger) -> None:
    """
    Detailed dump (snake-style), without dumping the whole policy repr.

    PPO-like:
      - features_extractor
      - mlp_extractor
      - action_net
      - value_net
    """
    policy = getattr(model, "policy", None)
    logger.info("Policy Network (detailed):")
    if policy is None:
        logger.info("  <no model.policy>")
        return

    logger.info(f"  policy_class: {policy.__class__.__name__}")

    # PPO/MaskablePPO branch
    feat = getattr(policy, "features_extractor", None)
    if feat is not None:
        _log_block(logger, "  features_extractor:", feat)

    mlp = getattr(policy, "mlp_extractor", None)
    if mlp is not None:
        _log_block(logger, "  mlp_extractor:", mlp)

    an = getattr(policy, "action_net", None)
    if an is not None:
        _log_block(logger, "  action_net:", an)

    vn = getattr(policy, "value_net", None)
    if vn is not None:
        _log_block(logger, "  value_net:", vn)


def log_model_params(*, model: Any, logger) -> None:
    """
    Same sections as snake:
      - summary (K/M/B)
      - detailed (per-submodule counts)
    """
    logger.info("Model params (summary):")
    logger.info(format_sb3_param_summary(model))
    logger.info("Model params (detailed):")
    logger.info(format_sb3_param_report(model))


__all__ = [
    "log_runtime_info",
    "log_env_reward_summary",
    "log_ppo_params",
    "log_model_params",
    "log_policy_compact",
    "log_policy_full",
]
