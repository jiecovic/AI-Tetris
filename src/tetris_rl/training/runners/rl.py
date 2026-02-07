# src/tetris_rl/training/runners/rl.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from tetris_rl.config.io import to_plain_dict
from tetris_rl.config.root import ExperimentConfig
from tetris_rl.runs.checkpoint_manifest import (
    CheckpointEntry,
    CheckpointManifest,
    save_checkpoint_manifest,
    update_checkpoint_manifest,
)
from tetris_rl.runs.run_io import make_run_paths, materialize_run_paths
from tetris_rl.runs.run_manifest import write_run_manifest
from tetris_rl.training.callbacks.eval_checkpoint import (
    EvalCheckpointCallback,
    EvalCheckpointSpec,
)
from tetris_rl.training.callbacks.info_logger import InfoLoggerCallback
from tetris_rl.training.callbacks.latest_checkpoint import (
    LatestCheckpointCallback,
    LatestCheckpointSpec,
)
from tetris_rl.training.env_factory import make_vec_env_from_cfg
from tetris_rl.training.model_factory import make_model_from_cfg
from tetris_rl.training.reporting import (
    log_policy_compact,
    log_policy_full,
    log_ppo_params,
    log_runtime_info,
)
from tetris_rl.training.runners.imitation import run_imitation_phase
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.paths import repo_root as find_repo_root


def _with_env_cfg(*, cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["env"] = dict(env_cfg)
    return out


def _resolve_resume_checkpoint(*, resume: str) -> Path:
    """
    Resolve a resume target.

    Project semantics:
      - If `resume` points to a run directory, resume from:
          <resume>/checkpoints/latest.zip
      - If `resume` points directly to a .zip file, use it as-is.
    """
    p = Path(resume)
    if p.suffix.lower() == ".zip":
        return p

    # Treat as run dir (as requested: cnn_run_013 -> always checkpoints/latest.zip)
    return p / "checkpoints" / "latest.zip"


def run_rl_experiment(cfg: DictConfig) -> int:
    cfg_dict = to_plain_dict(cfg)
    exp_cfg = ExperimentConfig.model_validate(cfg_dict)

    run_cfg = exp_cfg.run
    learn_cfg = exp_cfg.learn
    algo_cfg = exp_cfg.algo
    checkpoints_cfg = exp_cfg.checkpoints
    eval_cfg = exp_cfg.eval
    imitation_cfg = exp_cfg.imitation
    env_train_cfg = exp_cfg.env_train
    env_eval_cfg = exp_cfg.env_eval
    policy_cfg = exp_cfg.policy

    paths = make_run_paths(run_cfg=run_cfg)

    logger = setup_logger(name="tetris_rl.train", use_rich=True, level=str(exp_cfg.log_level))
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

    # ==============================================================
    # Model (fresh or resume)
    # ==============================================================
    t0 = time.perf_counter()
    resume_target = getattr(learn_cfg, "resume", None)
    is_resume = bool(resume_target and str(resume_target).strip())

    if is_resume:
        resume_ckpt = _resolve_resume_checkpoint(resume=str(resume_target).strip())
        logger.info(f"[resume] target={resume_target!r}")
        logger.info(f"[resume] ckpt={resume_ckpt.resolve() if resume_ckpt.exists() else resume_ckpt}")

        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f"resume checkpoint not found: {resume_ckpt} "
                f"(from learn.resume={resume_target!r})"
            )

        algo_type = str(algo_cfg.type).strip().lower()
        device = str(run_cfg.device).strip() or "auto"

        logger.info("[timing] loading model from checkpoint...")
        if algo_type == "maskable_ppo":
            from sb3_contrib.ppo_mask import MaskablePPO

            model = MaskablePPO.load(
                path=str(resume_ckpt),
                env=built.vec_env,
                device=device,
            )
            model._tetris_algo_type = "maskable_ppo"
        elif algo_type == "ppo":
            from stable_baselines3 import PPO

            model = PPO.load(
                path=str(resume_ckpt),
                env=built.vec_env,
                device=device,
            )
            model._tetris_algo_type = "ppo"
        elif algo_type == "dqn":
            from stable_baselines3 import DQN

            model = DQN.load(
                path=str(resume_ckpt),
                env=built.vec_env,
                device=device,
            )
            model._tetris_algo_type = "dqn"
        else:
            raise ValueError(f"unsupported algo type for resume: {algo_type!r}")

        logger.info(f"[timing] model loaded: {time.perf_counter() - t0:.2f}s")
    else:
        logger.info("[timing] building model...")
        model = make_model_from_cfg(
            cfg=policy_cfg,
            algo_cfg=algo_cfg,
            run_cfg=run_cfg,
            vec_env=built.vec_env,
            tensorboard_log=paths.tb_dir,
        )
        logger.info(f"[timing] model built: {time.perf_counter() - t0:.2f}s")

    logger.info(f"[train] run_dir={paths.run_dir.resolve()}")
    logger.info(f"[train] tb_dir={(paths.tb_dir.resolve() if paths.tb_dir is not None else None)}")
    logger.info(f"[train] checkpoints={(paths.ckpt_dir.resolve())}")
    logger.info(f"[train] obs_space={built.vec_env.observation_space}")
    logger.info(f"[train] action_space={built.vec_env.action_space}")

    # --- token count debug (outside the model) ---
    fe = getattr(model.policy, "features_extractor", None)
    if fe is None:
        fe = getattr(model.policy, "pi_features_extractor", None)

    tok = getattr(fe, "tokenizer", None) if fe is not None else None
    mix = getattr(fe, "token_mixer", None) if fe is not None else None

    if tok is not None and hasattr(tok, "stream_spec"):
        spec = tok.stream_spec()
        logger.info(
            "[tokens] stream_T=%d d_model=%d board_T=%s special_T=%s",
            int(spec.T),
            int(spec.d_model),
            getattr(tok, "n_board_tokens", None),
            getattr(tok, "n_special_tokens", None),
        )

    # CLS-adjusted total (what attention sees)
    if tok is not None and mix is not None:
        use_cls = getattr(getattr(mix, "spec", None), "use_cls", None)
        num_cls = getattr(getattr(mix, "spec", None), "num_cls_tokens", None)
        logger.info("[tokens] mixer_use_cls=%s mixer_num_cls=%s", use_cls, num_cls)

    log_runtime_info(logger=logger)

    # algo tag set by model_factory (or by resume loader above)
    algo_type = str(getattr(model, "_tetris_algo_type", "")).strip().lower()

    # Only PPO-like algos have PPO params to log
    if algo_type in {"ppo", "maskable_ppo"}:
        log_ppo_params(model=model, logger=logger, tb_log=paths.tb_dir)
    else:
        logger.info(f"[train] algo={algo_type or type(model).__name__} (skipping PPO param log)")

    log_policy_compact(model=model, logger=logger)
    if str(exp_cfg.log_level).lower() in {"debug", "trace"}:
        log_policy_full(model=model, logger=logger)

    cb_verbose = 1  # keep existing behavior

    # ==============================================================
    # Imitation phase (optional)
    # ==============================================================
    run_imitation_phase(
        cfg=cfg_train,
        model=model,
        imitation_cfg=imitation_cfg,
        eval_cfg=eval_cfg,
        checkpoints_cfg=checkpoints_cfg,
        run_cfg=run_cfg,
        run_dir=paths.run_dir,
        repo=find_repo_root(),
        logger=logger,
        algo_type=algo_type,
    )

    # ==============================================================
    # Callbacks (RL phase only)
    # ==============================================================
    callbacks: list[BaseCallback] = [
        InfoLoggerCallback(cfg=cfg_train, verbose=0),
        LatestCheckpointCallback(
            cfg=cfg_train,
            spec=LatestCheckpointSpec(
                checkpoint_dir=paths.ckpt_dir,
                latest_every=int(checkpoints_cfg.latest_every),
                verbose=0,
            ),
        ),
    ]

    if int(eval_cfg.eval_every) > 0 and str(eval_cfg.mode).strip().lower() != "off":
        # IMPORTANT: keep EvalCheckpointSpec unchanged by passing an eval-specific cfg
        # into the callback (callback still receives `cfg` and can build eval env as before).
        eval_cfg = _with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg.model_dump(mode="json"))

        callbacks.append(
            EvalCheckpointCallback(
                cfg=eval_cfg,
                spec=EvalCheckpointSpec(
                    checkpoint_dir=paths.ckpt_dir,
                    eval_every=int(eval_cfg.eval_every),
                    eval=eval_cfg,
                    base_seed=int(run_cfg.seed),
                    run_cfg=run_cfg,
                    verbose=cb_verbose,
                ),
            )
        )
    else:
        logger.info("[eval] disabled (eval.mode=off or eval.eval_every<=0)")

    cb_list = CallbackList(callbacks)

    if int(learn_cfg.total_timesteps) > 0:
        model.learn(
            total_timesteps=int(learn_cfg.total_timesteps),
            callback=cb_list,
            progress_bar=True,
            reset_num_timesteps=not is_resume,
        )
        final_path = paths.ckpt_dir / "final.zip"
        model.save(str(final_path))
        final_steps = int(getattr(model, "num_timesteps", int(learn_cfg.total_timesteps)))
        update_checkpoint_manifest(
            manifest_path=paths.ckpt_dir / "manifest.json",
            field="final",
            entry=CheckpointEntry(
                path=final_path.name,
                timesteps=final_steps,
            ),
        )
    else:
        logger.info("[rl] skipped (learn.total_timesteps <= 0)")

    built.vec_env.close()

    logger.info("[done]")
    return 0


__all__ = ["run_rl_experiment"]
