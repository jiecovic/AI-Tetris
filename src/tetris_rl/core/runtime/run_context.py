# src/tetris_rl/core/runtime/run_context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tetris_rl.core.agents.expert import make_expert_policy
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.runs.checkpoints.checkpoint_poll import CheckpointPoller
from tetris_rl.core.runs.run_resolver import (
    InferenceArtifact,
    RunSpec,
    load_ga_policy_from_artifact,
    load_run_spec,
    resolve_env_cfg,
    resolve_inference_artifact,
)
from tetris_rl.core.training.model_io import load_model_from_algo_config, warn_if_maskable_with_multidiscrete


@dataclass(frozen=True)
class RunContext:
    spec: RunSpec
    artifact: InferenceArtifact
    env: Any
    game: Any
    ckpt: Any
    algo_type: str
    model: Any | None
    ga_policy: Any | None
    expert_policy: Any | None
    poller: CheckpointPoller | None


def _with_env_cfg(*, cfg: dict[str, Any], env_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["env"] = dict(env_cfg)
    return out


def _apply_piece_rule_override(cfg: dict[str, Any], piece_rule: str | None) -> dict[str, Any]:
    if piece_rule is None:
        return cfg
    cfg_out = dict(cfg)
    cfg_env = cfg_out.get("env", {}) or {}
    if not isinstance(cfg_env, dict):
        cfg_env = {}
    game_cfg = cfg_env.get("game", {}) or {}
    if not isinstance(game_cfg, dict):
        game_cfg = {}
    game_cfg = dict(game_cfg)
    game_cfg["piece_rule"] = str(piece_rule).strip().lower()
    cfg_env = dict(cfg_env)
    cfg_env["game"] = game_cfg
    cfg_out["env"] = cfg_env
    return cfg_out


def build_run_context(
    *,
    run: str,
    which: str,
    which_env: str,
    seed: int,
    device: str,
    piece_rule: str | None,
    reload_every_s: float,
    use_expert: bool,
    random_action: bool,
    expert_args: Any,
) -> RunContext:
    spec = load_run_spec(run)
    artifact = resolve_inference_artifact(spec=spec, which=str(which))

    env_cfg = resolve_env_cfg(spec=spec, which_env=str(which_env))
    cfg_ctx = _with_env_cfg(cfg=spec.cfg_plain, env_cfg=env_cfg)
    cfg_ctx = _apply_piece_rule_override(cfg_ctx, piece_rule)

    built = make_env_from_cfg(cfg=cfg_ctx, seed=int(seed))
    env = built.env

    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env must expose .game (rust engine wrapper) for runtime tools")

    algo_type = str(spec.algo_type)
    expert_policy: Optional[Any] = None
    if bool(use_expert):
        expert_policy = make_expert_policy(args=expert_args, engine=game)

    ckpt = artifact.path
    ga_policy = None
    if spec.algo_type == "ga":
        ga_policy = load_ga_policy_from_artifact(spec=spec, artifact=artifact, env=env)

    model = None
    if (not bool(use_expert)) and (not bool(random_action)):
        if spec.algo_type != "ga":
            loaded = load_model_from_algo_config(algo_cfg=spec.algo_cfg, ckpt=ckpt, device=str(device))
            model = loaded.model
            algo_type = loaded.algo_type
            ckpt = loaded.ckpt
            if algo_type == "maskable_ppo":
                warn_if_maskable_with_multidiscrete(algo_cfg=spec.algo_cfg, env=env)

    poller = None
    if spec.algo_type != "ga" and model is not None and float(reload_every_s) > 0.0:
        poller = CheckpointPoller(
            run_dir=spec.run_dir,
            which=str(which),
            algo_cfg=spec.algo_cfg,
            device=str(device),
            reload_every_s=float(reload_every_s),
        )
        poller.set_current(ckpt=ckpt, model=model, algo_type=str(algo_type))

    return RunContext(
        spec=spec,
        artifact=artifact,
        env=env,
        game=game,
        ckpt=ckpt,
        algo_type=algo_type,
        model=model,
        ga_policy=ga_policy,
        expert_policy=expert_policy,
        poller=poller,
    )


__all__ = ["RunContext", "build_run_context"]
