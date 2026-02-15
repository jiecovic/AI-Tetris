# src/tetris_rl/core/training/runners/imitation.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, cast

from omegaconf import DictConfig

from tetris_rl.core.config.io import to_plain_dict
from tetris_rl.core.config.root import ImitationExperimentConfig
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.runs.checkpoints.checkpoint_manifest import CheckpointEntry, update_checkpoint_manifest
from tetris_rl.core.training.config import AlgoConfig
from tetris_rl.core.training.env_factory import make_vec_env_from_cfg
from tetris_rl.core.training.imitation.algorithm import ImitationAlgorithm
from tetris_rl.core.training.model_factory import build_policy_from_cfg
from tetris_rl.core.training.model_io import load_model_from_algo_config, try_load_policy_checkpoint
from tetris_rl.core.training.reporting import (
    log_env_reward_summary,
    log_policy_compact,
    log_policy_full,
    log_runtime_info,
)
from tetris_rl.core.training.runners.common import (
    ensure_checkpoint_manifest,
    init_run_artifacts,
    with_env_cfg,
)
from tetris_rl.core.training.tb_logger import maybe_tb_logger
from tetris_rl.core.utils.logging import setup_logger
from tetris_rl.core.utils.paths import repo_root as find_repo_root
from tetris_rl.core.utils.seed import seed_all


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
    seed_all(int(run_cfg.seed))

    logger = setup_logger(name="tetris_rl.core.imitation", use_rich=True, level=str(exp_cfg.log_level))
    artifacts = init_run_artifacts(cfg_dict=cfg_dict, run_cfg=run_cfg, logger=logger)
    paths = artifacts.paths
    logger.info(f"[run] n_envs={run_cfg.n_envs} vec={run_cfg.vec} device={run_cfg.device}")
    _ = ensure_checkpoint_manifest(ckpt_dir=paths.ckpt_dir)

    t0 = time.perf_counter()
    logger.info("[timing] building vec env...")
    cfg_train = with_env_cfg(cfg=cfg_dict, env_cfg=env_train_cfg.model_dump(mode="json"))
    probe_train = make_env_from_cfg(cfg=cfg_train, seed=int(run_cfg.seed))
    log_env_reward_summary(
        logger=logger,
        label="train",
        built_env=probe_train,
        env_cfg=env_train_cfg.model_dump(mode="json"),
        time_limit_steps=run_cfg.max_episode_steps,
    )
    try:
        probe_train.env.close()
    except Exception:
        pass
    built = make_vec_env_from_cfg(cfg=cfg_train, run_cfg=run_cfg)
    logger.info(f"[timing] vec env built: {time.perf_counter() - t0:.2f}s")

    if str(algo_cfg.type).strip().lower() != "imitation":
        raise ValueError(f"imitation algo type must be 'imitation' (got {algo_cfg.type!r})")

    policy_backend = str(imitation_params.policy_backend).strip().lower()
    if policy_backend not in {"ppo", "maskable_ppo"}:
        raise ValueError(f"imitation policy_backend must be 'ppo' or 'maskable_ppo' (got {policy_backend!r})")
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
            algo_type = cast(Literal["ppo", "maskable_ppo"], policy_backend)
            loaded = load_model_from_algo_config(
                algo_cfg=AlgoConfig(type=algo_type),
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
    log_runtime_info(logger=logger)
    log_policy_compact(model=model, logger=logger)
    # Imitation training is typically run with small policy variants; keep full layer dump visible.
    log_policy_full(model=model, logger=logger)

    cfg_eval = with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg.model_dump(mode="json"))
    probe_eval = make_env_from_cfg(cfg=cfg_eval, seed=int(run_cfg.seed))
    log_env_reward_summary(
        logger=logger,
        label="eval",
        built_env=probe_eval,
        env_cfg=env_eval_cfg.model_dump(mode="json"),
        time_limit_steps=None,
    )
    try:
        probe_eval.env.close()
    except Exception:
        pass
    tb_logger = maybe_tb_logger(paths.tb_dir)
    model.learn(
        cfg=cfg_eval,
        learn_cfg=learn_cfg,
        callbacks_cfg=callbacks_cfg,
        run_cfg=run_cfg,
        run_dir=paths.run_dir,
        repo=find_repo_root(),
        logger=logger,
        tb_logger=tb_logger,
    )

    final_path = paths.ckpt_dir / "final.zip"
    model.save(str(final_path))
    update_checkpoint_manifest(
        manifest_path=paths.ckpt_dir / "manifest.json",
        field="final",
        entry=CheckpointEntry(path=final_path.name, timesteps=0),
    )

    built.vec_env.close()
    if tb_logger is not None:
        tb_logger.close()
    logger.info("[done]")
    return 0


__all__ = ["run_imitation_experiment"]
