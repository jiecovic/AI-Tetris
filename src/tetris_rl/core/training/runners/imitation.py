# src/tetris_rl/core/training/runners/imitation.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tetris_rl.core.config.io import to_plain_dict
from tetris_rl.core.config.root import ImitationExperimentConfig
from tetris_rl.core.runs.checkpoints.checkpoint_manifest import (
    CheckpointEntry,
    CheckpointManifest,
    save_checkpoint_manifest,
    update_checkpoint_manifest,
)
from tetris_rl.core.runs.run_io import make_run_paths, materialize_run_paths
from tetris_rl.core.runs.run_manifest import write_run_manifest
from tetris_rl.core.training.env_factory import make_vec_env_from_cfg
from tetris_rl.core.training.config import AlgoConfig
from tetris_rl.core.training.imitation.algorithm import ImitationAlgorithm
from tetris_rl.core.training.model_factory import build_policy_from_cfg
from tetris_rl.core.training.model_io import load_model_from_algo_config, try_load_policy_checkpoint
from tetris_rl.core.utils.logging import setup_logger
from tetris_rl.core.utils.paths import repo_root as find_repo_root


def _with_env_cfg(*, cfg: dict[str, Any], env_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["env"] = dict(env_cfg)
    return out


def run_imitation_experiment(cfg: DictConfig) -> int:
    cfg_dict = to_plain_dict(cfg)
    exp_cfg = ImitationExperimentConfig.model_validate(cfg_dict)

    run_cfg = exp_cfg.run
    env_train_cfg = exp_cfg.env_train
    env_eval_cfg = exp_cfg.env_eval
    policy_cfg = exp_cfg.policy
    algo_cfg = exp_cfg.algo
    callbacks_cfg = exp_cfg.callbacks
    learn_cfg = exp_cfg.learn
    imitation_params = exp_cfg.algo.params

    paths = make_run_paths(run_cfg=run_cfg)

    logger = setup_logger(name="tetris_rl.core.imitation", use_rich=True, level=str(exp_cfg.log_level))
    logger.info(f"[run] dir: {paths.run_dir}")
    logger.info(f"[run] n_envs={run_cfg.n_envs} vec={run_cfg.vec} device={run_cfg.device}")

    t0 = time.perf_counter()
    materialize_run_paths(paths=paths)
    config_path = paths.run_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(cfg_dict), f=config_path)
    write_run_manifest(run_dir=paths.run_dir, config_path=config_path)
    manifest_path = paths.ckpt_dir / "manifest.json"
    if not manifest_path.exists():
        save_checkpoint_manifest(manifest_path, CheckpointManifest())
    logger.info(f"[timing] paths+snapshot: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    logger.info("[timing] building vec env...")
    cfg_train = _with_env_cfg(cfg=cfg_dict, env_cfg=env_train_cfg.model_dump(mode="json"))
    built = make_vec_env_from_cfg(cfg=cfg_train, run_cfg=run_cfg)
    logger.info(f"[timing] vec env built: {time.perf_counter() - t0:.2f}s")

    if str(algo_cfg.type).strip().lower() != "imitation":
        raise ValueError(f"imitation algo type must be 'imitation' (got {algo_cfg.type!r})")

    policy_backend = str(imitation_params.policy_backend).strip().lower()
    logger.info("[timing] building policy (%s)...", policy_backend)
    policy = build_policy_from_cfg(
        policy_cfg=policy_cfg,
        policy_backend=policy_backend,
        observation_space=built.vec_env.observation_space,
        action_space=built.vec_env.action_space,
        device=str(run_cfg.device).strip() or "cpu",
    )
    model = ImitationAlgorithm(
        policy=policy,
        env=built.vec_env,
        params=imitation_params,
    )
    logger.info(f"[timing] policy built: {time.perf_counter() - t0:.2f}s")

    resume_target = getattr(learn_cfg, "resume", None)
    if resume_target:
        resume_path = Path(str(resume_target)).expanduser()
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoints" / "latest.zip"
        if not resume_path.exists():
            raise FileNotFoundError(f"imitation resume checkpoint not found: {resume_path}")
        loaded_policy = try_load_policy_checkpoint(resume_path, device=str(run_cfg.device))
        if loaded_policy is not None:
            try:
                model.policy.load_state_dict(loaded_policy.state_dict(), strict=True)
                logger.info(f"[resume] loaded imitation weights: {resume_path}")
            except Exception as e:
                raise RuntimeError(f"failed to load imitation weights from {resume_path}") from e
        else:
            loaded = load_model_from_algo_config(
                algo_cfg=AlgoConfig(type=policy_backend),
                ckpt=resume_path,
                device=str(run_cfg.device),
            )
            try:
                model.policy.load_state_dict(loaded.model.policy.state_dict(), strict=True)
                logger.info(f"[resume] loaded sb3 weights: {resume_path}")
            except Exception as e:
                raise RuntimeError(f"failed to load sb3 weights from {resume_path}") from e

    logger.info(f"[train] run_dir={paths.run_dir.resolve()}")
    logger.info(f"[train] tb_dir={(paths.tb_dir.resolve() if paths.tb_dir is not None else None)}")
    logger.info(f"[train] checkpoints={(paths.ckpt_dir.resolve())}")
    logger.info(f"[train] obs_space={built.vec_env.observation_space}")
    logger.info(f"[train] action_space={built.vec_env.action_space}")

    cfg_eval = _with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg.model_dump(mode="json"))
    model.learn(
        cfg=cfg_eval,
        learn_cfg=learn_cfg,
        callbacks_cfg=callbacks_cfg,
        run_cfg=run_cfg,
        run_dir=paths.run_dir,
        repo=find_repo_root(),
        logger=logger,
    )

    final_path = paths.ckpt_dir / "final.zip"
    model.save(str(final_path))
    update_checkpoint_manifest(
        manifest_path=paths.ckpt_dir / "manifest.json",
        field="final",
        entry=CheckpointEntry(path=final_path.name, timesteps=0),
    )

    built.vec_env.close()
    logger.info("[done]")
    return 0


__all__ = ["run_imitation_experiment"]
