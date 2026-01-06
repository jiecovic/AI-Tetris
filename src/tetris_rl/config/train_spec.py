# src/tetris_rl/config/train_spec.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from tetris_rl.config.schema_types import (
    get_bool,
    get_float,
    get_int,
    get_mapping,
    get_str,
    require_mapping,
    require_mapping_strict,
)


@dataclass(frozen=True)
class TrainCheckpointsSpec:
    """
    Checkpoint cadence for training.

    Semantics:
      - latest_every: save checkpoints/latest.zip every N environment steps.
    """

    latest_every: int = 200_000


@dataclass(frozen=True)
class TrainEvalSpec:
    """
    Training-time evaluation semantics.

    This is a training hook (for TB + best checkpoints), not a benchmarking suite.

    mode:
      - "off": disable evaluation entirely
      - "rl": evaluate only during RL phase
      - "imitation": evaluate only during imitation phase
      - "both": evaluate during both phases

    eval_every:
      - <= 0 disables evaluation (regardless of mode)
    """

    mode: str = "off"  # off | rl | imitation | both
    steps: int = 100_000
    eval_every: int = 0

    deterministic: bool = True
    seed_offset: int = 10_000
    num_envs: int = 1

    # Optional config patch applied ONLY when building the eval environment.
    # Intended usage: disable warmup, remove max_steps truncation, etc.
    env_override: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImitationSpec:
    enabled: bool = False

    # offline dataset (required when enabled)
    dataset_dir: str = ""

    # training
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    shuffle: bool = True

    # optional: limit how many samples to use (0 = all)
    max_samples: int = 0

    # archive (copy of latest.zip)
    save_archive: bool = True
    archive_dir: str = "checkpoints/imitation"


@dataclass(frozen=True)
class RLAlgoSpec:
    type: str = "ppo"  # ppo | maskable_ppo | dqn
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RLSpec:
    enabled: bool = True
    total_timesteps: int = 200_000
    algo: RLAlgoSpec = field(default_factory=RLAlgoSpec)


@dataclass(frozen=True)
class TrainSpec:
    checkpoints: TrainCheckpointsSpec = field(default_factory=TrainCheckpointsSpec)
    eval: TrainEvalSpec = field(default_factory=TrainEvalSpec)
    imitation: ImitationSpec = field(default_factory=ImitationSpec)
    rl: RLSpec = field(default_factory=RLSpec)


def parse_train_spec(*, cfg: Dict[str, Any]) -> TrainSpec:
    """
    Canonical layout ONLY:

      train:
        checkpoints: { latest_every }
        eval:        { mode, steps, eval_every, deterministic, seed_offset, num_envs, env_override }
        imitation:   { enabled, dataset_dir, epochs, batch_size, learning_rate, max_grad_norm, shuffle, max_samples, ... }
        rl:          { enabled, total_timesteps, algo: { type, params } }
    """
    root = require_mapping(cfg, where="cfg")
    train = require_mapping_strict(root.get("train", None), where="cfg.train")

    # ---------------------------
    # checkpoints
    # ---------------------------
    ckpt = get_mapping(train, "checkpoints", default={}, where="cfg.train.checkpoints")

    latest_every = max(
        1,
        get_int(
            ckpt,
            "latest_every",
            default=TrainCheckpointsSpec.latest_every,
            where="cfg.train.checkpoints.latest_every",
        ),
    )
    checkpoints = TrainCheckpointsSpec(latest_every=int(latest_every))

    # ---------------------------
    # eval (training-time)
    # ---------------------------
    ev = get_mapping(train, "eval", default={}, where="cfg.train.eval")

    mode = get_str(ev, "mode", default=TrainEvalSpec.mode, where="cfg.train.eval.mode").strip().lower()
    if mode not in {"off", "rl", "imitation", "both"}:
        raise ValueError(f"cfg.train.eval.mode must be off|rl|imitation|both, got {mode!r}")

    steps = max(
        1,
        get_int(
            ev,
            "steps",
            default=TrainEvalSpec.steps,
            where="cfg.train.eval.steps",
        ),
    )
    eval_every = max(
        0,
        get_int(
            ev,
            "eval_every",
            default=TrainEvalSpec.eval_every,
            where="cfg.train.eval.eval_every",
        ),
    )
    deterministic = get_bool(
        ev,
        "deterministic",
        default=TrainEvalSpec.deterministic,
        where="cfg.train.eval.deterministic",
    )
    seed_offset = get_int(
        ev,
        "seed_offset",
        default=TrainEvalSpec.seed_offset,
        where="cfg.train.eval.seed_offset",
    )
    num_envs = max(
        1,
        get_int(
            ev,
            "num_envs",
            default=TrainEvalSpec.num_envs,
            where="cfg.train.eval.num_envs",
        ),
    )

    env_override = get_mapping(ev, "env_override", default={}, where="cfg.train.eval.env_override")

    eval_spec = TrainEvalSpec(
        mode=str(mode),
        steps=int(steps),
        eval_every=int(eval_every),
        deterministic=bool(deterministic),
        seed_offset=int(seed_offset),
        num_envs=int(num_envs),
        env_override=dict(env_override),
    )

    # ---------------------------
    # imitation
    # ---------------------------
    im = get_mapping(train, "imitation", default={}, where="cfg.train.imitation")

    imitation = ImitationSpec(
        enabled=get_bool(im, "enabled", default=ImitationSpec.enabled, where="cfg.train.imitation.enabled"),
        dataset_dir=get_str(im, "dataset_dir", default=ImitationSpec.dataset_dir, where="cfg.train.imitation.dataset_dir"),
        epochs=max(1, get_int(im, "epochs", default=ImitationSpec.epochs, where="cfg.train.imitation.epochs")),
        batch_size=max(1, get_int(im, "batch_size", default=ImitationSpec.batch_size, where="cfg.train.imitation.batch_size")),
        learning_rate=get_float(im, "learning_rate", default=ImitationSpec.learning_rate, where="cfg.train.imitation.learning_rate"),
        max_grad_norm=get_float(im, "max_grad_norm", default=ImitationSpec.max_grad_norm, where="cfg.train.imitation.max_grad_norm"),
        shuffle=get_bool(im, "shuffle", default=ImitationSpec.shuffle, where="cfg.train.imitation.shuffle"),
        max_samples=max(0, get_int(im, "max_samples", default=ImitationSpec.max_samples, where="cfg.train.imitation.max_samples")),
        save_archive=get_bool(im, "save_archive", default=ImitationSpec.save_archive, where="cfg.train.imitation.save_archive"),
        archive_dir=get_str(im, "archive_dir", default=ImitationSpec.archive_dir, where="cfg.train.imitation.archive_dir"),
    )

    if imitation.enabled and not imitation.dataset_dir.strip():
        raise ValueError("cfg.train.imitation.enabled=true requires cfg.train.imitation.dataset_dir to be set")

    # ---------------------------
    # rl
    # ---------------------------
    rl_obj = get_mapping(train, "rl", default={}, where="cfg.train.rl")

    total_timesteps = max(
        0,
        get_int(rl_obj, "total_timesteps", default=RLSpec.total_timesteps, where="cfg.train.rl.total_timesteps"),
    )

    algo_obj = require_mapping(rl_obj.get("algo", None), where="cfg.train.rl.algo")
    algo_type = get_str(algo_obj, "type", default=RLAlgoSpec.type, where="cfg.train.rl.algo.type").strip().lower()
    if algo_type not in {"ppo", "maskable_ppo", "dqn"}:
        raise ValueError(f"cfg.train.rl.algo.type must be 'ppo'|'maskable_ppo'|'dqn', got {algo_type!r}")

    algo_params = algo_obj.get("params", {}) or {}
    algo_params = require_mapping(algo_params, where="cfg.train.rl.algo.params")

    rl = RLSpec(
        enabled=get_bool(rl_obj, "enabled", default=RLSpec.enabled, where="cfg.train.rl.enabled"),
        total_timesteps=int(total_timesteps),
        algo=RLAlgoSpec(type=str(algo_type), params=dict(algo_params)),
    )

    return TrainSpec(checkpoints=checkpoints, eval=eval_spec, imitation=imitation, rl=rl)


__all__ = [
    "TrainSpec",
    "TrainCheckpointsSpec",
    "TrainEvalSpec",
    "ImitationSpec",
    "RLSpec",
    "RLAlgoSpec",
    "parse_train_spec",
]
