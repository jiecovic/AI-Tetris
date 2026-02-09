# src/tetris_rl/core/training/runners/ppo.py
from __future__ import annotations

import time

from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.core.callbacks import EvalCallback, InfoLoggerCallback, LatestCallback, SB3CallbackAdapter
from tetris_rl.core.config.io import to_plain_dict
from tetris_rl.core.config.root import ExperimentConfig
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.runs.checkpoints.checkpoint_manifest import (
    CheckpointEntry,
    update_checkpoint_manifest,
)
from tetris_rl.core.runs.run_resolver import resolve_resume_checkpoint
from tetris_rl.core.training.env_factory import make_vec_env_from_cfg
from tetris_rl.core.training.evaluation.eval_checkpoint_core import EvalCheckpointCoreSpec
from tetris_rl.core.training.evaluation.latest_checkpoint_core import LatestCheckpointCoreSpec
from tetris_rl.core.training.model_factory import make_model_from_cfg
from tetris_rl.core.training.model_io import load_model_from_algo_config, try_load_policy_checkpoint
from tetris_rl.core.training.reporting import (
    log_env_reward_summary,
    log_policy_compact,
    log_policy_full,
    log_ppo_params,
    log_runtime_info,
)
from tetris_rl.core.training.runners.common import (
    ensure_checkpoint_manifest,
    init_run_artifacts,
    with_env_cfg,
)
from tetris_rl.core.utils.logging import setup_logger
from tetris_rl.core.utils.seed import seed_all


def run_ppo_experiment(cfg: DictConfig) -> int:
    cfg_dict = to_plain_dict(cfg)
    exp_cfg = ExperimentConfig.model_validate(cfg_dict)

    run_cfg = exp_cfg.run
    learn_cfg = exp_cfg.learn
    algo_cfg = exp_cfg.algo
    callbacks_cfg = exp_cfg.callbacks
    eval_cfg = callbacks_cfg.eval_checkpoint
    env_train_cfg = exp_cfg.env_train
    env_eval_cfg = exp_cfg.env_eval
    policy_cfg = exp_cfg.policy
    seed_all(int(run_cfg.seed))
    max_steps_per_episode = getattr(learn_cfg, "max_steps_per_episode", None)

    logger = setup_logger(name="tetris_rl.core.training", use_rich=True, level=str(exp_cfg.log_level))
    artifacts = init_run_artifacts(cfg_dict=cfg_dict, run_cfg=run_cfg, logger=logger)
    paths = artifacts.paths
    logger.info(f"[run] n_envs={run_cfg.n_envs} vec={run_cfg.vec} device={run_cfg.device}")
    _ = ensure_checkpoint_manifest(ckpt_dir=paths.ckpt_dir)

    t0 = time.perf_counter()
    logger.info("[timing] building vec env...")
    cfg_train = with_env_cfg(
        cfg=cfg_dict,
        env_cfg=env_train_cfg.model_dump(mode="json"),
        max_steps_per_episode=max_steps_per_episode,
    )
    probe_train = make_env_from_cfg(cfg=cfg_train, seed=int(run_cfg.seed))
    log_env_reward_summary(
        logger=logger,
        label="train",
        built_env=probe_train,
        env_cfg=env_train_cfg.model_dump(mode="json"),
        time_limit_steps=None,
    )
    try:
        probe_train.env.close()
    except Exception:
        pass
    built = make_vec_env_from_cfg(cfg=cfg_train, run_cfg=run_cfg)
    logger.info(f"[timing] vec env built: {time.perf_counter() - t0:.2f}s")

    # ==============================================================
    # Model (fresh or resume)
    # ==============================================================
    t0 = time.perf_counter()
    resume_target = getattr(learn_cfg, "resume", None)
    is_resume = bool(resume_target and str(resume_target).strip())
    imitation_resume = False

    if is_resume:
        resume_ckpt = resolve_resume_checkpoint(str(resume_target).strip())
        logger.info(f"[resume] target={resume_target!r}")
        logger.info(f"[resume] ckpt={resume_ckpt.resolve() if resume_ckpt.exists() else resume_ckpt}")

        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f"resume checkpoint not found: {resume_ckpt} "
                f"(from learn.resume={resume_target!r})"
            )

        algo_type = str(algo_cfg.type).strip().lower()
        device = str(run_cfg.device).strip() or "auto"

        loaded_policy = try_load_policy_checkpoint(str(resume_ckpt), device=str(device))
        if loaded_policy is not None:
            imitation_resume = True
            logger.info("[timing] building model for imitation resume...")
            model = make_model_from_cfg(
                cfg=policy_cfg,
                algo_cfg=algo_cfg,
                run_cfg=run_cfg,
                vec_env=built.vec_env,
                tensorboard_log=paths.tb_dir,
            )
            try:
                model.policy.load_state_dict(loaded_policy.state_dict(), strict=True)
            except Exception as e:
                raise RuntimeError(f"failed to load imitation weights from {resume_ckpt}") from e
            model._tetris_algo_type = algo_type
            logger.info("[resume] loaded imitation weights (timesteps reset)")
            logger.info(f"[timing] model initialized: {time.perf_counter() - t0:.2f}s")
        else:
            logger.info("[timing] loading model from checkpoint...")
            loaded = load_model_from_algo_config(
                algo_cfg=algo_cfg,
                ckpt=resume_ckpt,
                device=device,
                env=built.vec_env,
            )
            model = loaded.model
            model._tetris_algo_type = str(loaded.algo_type)
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
    # Callbacks
    # ==============================================================
    callbacks: list[BaseCallback] = [
        InfoLoggerCallback(cfg=cfg_train, verbose=0),
    ]

    core_callbacks = []

    if bool(callbacks_cfg.latest.enabled) and int(callbacks_cfg.latest.every) > 0:
        core_callbacks.append(
            LatestCallback(
                spec=LatestCheckpointCoreSpec(
                    checkpoint_dir=paths.ckpt_dir,
                    latest_every=int(callbacks_cfg.latest.every),
                    verbose=0,
                ),
                event="step",
                progress_key="num_timesteps",
            )
        )

    if bool(callbacks_cfg.eval_checkpoint.enabled) and int(callbacks_cfg.eval_checkpoint.every) > 0:
        # Provide eval-specific cfg dict for env wiring.
        eval_cfg_plain = with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg.model_dump(mode="json"))
        probe_eval = make_env_from_cfg(cfg=eval_cfg_plain, seed=int(run_cfg.seed))
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

        eval_cb: EvalCallback | None = None

        def _emit(line: str) -> None:
            try:
                tqdm.write(line)
            except Exception:
                print(line, flush=True)

        def _log_scalar(name: str, value: float, _step: int) -> None:
            if eval_cb is None:
                return
            algo = eval_cb.algo
            if algo is None:
                return
            logger_ref = getattr(algo, "logger", None)
            if logger_ref is None:
                return
            try:
                logger_ref.record(str(name), float(value))
            except Exception:
                pass

        eval_cb = EvalCallback(
            spec=EvalCheckpointCoreSpec(
                checkpoint_dir=paths.ckpt_dir,
                eval_every=int(callbacks_cfg.eval_checkpoint.every),
                run_cfg=run_cfg,
                eval=eval_cfg,
                base_seed=int(run_cfg.seed),
                verbose=cb_verbose,
            ),
            cfg=eval_cfg_plain,
            event="step",
            progress_key="num_timesteps",
            phase=str(algo_cfg.type),
            emit=_emit,
            log_scalar=_log_scalar,
        )
        core_callbacks.append(eval_cb)
    else:
        logger.info("[eval] disabled (callbacks.eval_checkpoint disabled)")

    if core_callbacks:
        callbacks.append(SB3CallbackAdapter(core_callbacks))

    cb_list = CallbackList(callbacks)

    if int(learn_cfg.total_timesteps) > 0:
        model.learn(
            total_timesteps=int(learn_cfg.total_timesteps),
            callback=cb_list,
            progress_bar=True,
            reset_num_timesteps=(not is_resume) or imitation_resume,
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
        logger.info("[ppo] skipped (learn.total_timesteps <= 0)")

    built.vec_env.close()

    logger.info("[done]")
    return 0


__all__ = ["run_ppo_experiment"]
