# src/tetris_rl/cli/train.py
from __future__ import annotations

import argparse
from pathlib import Path
import time

from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from tetris_rl.config.resolve import resolve_config
from tetris_rl.config.run_spec import parse_run_spec
from tetris_rl.config.train_spec import parse_train_spec
from tetris_rl.config.snapshot import load_yaml, write_config_snapshot
from tetris_rl.runs.run_io import make_run_paths, materialize_run_paths
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
from tetris_rl.training.imitation.runner import run_imitation
from tetris_rl.training.model_factory import make_model_from_cfg
from tetris_rl.training.reporting import (
    log_policy_compact,
    log_policy_full,
    log_ppo_params,
    log_runtime_info,
)
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.paths import repo_root as find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/mlp.yaml")
    ap.add_argument("--log-level", type=str, default="info")
    args = ap.parse_args()

    repo_root = find_repo_root()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg = resolve_config(cfg=cfg, cfg_path=cfg_path)

    run = parse_run_spec(cfg=cfg)


    train = parse_train_spec(cfg=cfg)

    paths = make_run_paths(run_spec=run)

    logger = setup_logger(name="tetris_rl.train", use_rich=True, level=args.log_level)
    logger.info(f"[run] dir: {paths.run_dir}")
    logger.info(f"[run] n_envs={run.n_envs} vec={run.vec} device={run.device}")

    t0 = time.perf_counter()
    materialize_run_paths(paths=paths)
    write_config_snapshot(src_path=cfg_path, run_dir=paths.run_dir, resolved_cfg=cfg)
    logger.info(f"[timing] paths+snapshot: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    logger.info("[timing] building vec env...")
    built = make_vec_env_from_cfg(cfg=cfg, run_spec=run)
    logger.info(f"[timing] vec env built: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    logger.info("[timing] building model...")
    model = make_model_from_cfg(
        cfg=cfg,
        train_spec=train,
        run_spec=run,
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

    # algo tag set by model_factory
    algo_type = str(getattr(model, "_tetris_algo_type", "")).strip().lower()

    # Only PPO-like algos have PPO params to log
    if algo_type in {"ppo", "maskable_ppo"}:
        log_ppo_params(model=model, logger=logger, tb_log=paths.tb_dir)
    else:
        logger.info(f"[train] algo={algo_type or type(model).__name__} (skipping PPO param log)")

    log_policy_compact(model=model, logger=logger)
    if str(args.log_level).lower() in {"debug", "trace"}:
        log_policy_full(model=model, logger=logger)

    cb_verbose = 1  # keep existing behavior

    # ==============================================================
    # Imitation phase (optional)
    # ==============================================================
    if bool(train.imitation.enabled):
        if algo_type in {"ppo", "maskable_ppo"}:
            run_imitation(
                cfg=cfg,
                model=model,
                train_spec=train,
                run_spec=run,
                run_dir=paths.run_dir,
                repo=repo_root,
                logger=logger,
            )
        else:
            logger.info(
                f"[imitation] skipped (enabled=true but algo={algo_type!r} does not support imitation)"
            )
    else:
        logger.info("[imitation] skipped (train.imitation.enabled=false)")

    # ==============================================================
    # Callbacks (RL phase only)
    # ==============================================================

    callbacks: list[BaseCallback] = [
        InfoLoggerCallback(cfg=cfg, verbose=0),
        LatestCheckpointCallback(
            cfg=cfg,
            spec=LatestCheckpointSpec(
                checkpoint_dir=paths.ckpt_dir,
                latest_every=int(train.checkpoints.latest_every),
                verbose=0,
            ),
        ),
    ]

    if int(train.eval.eval_every) > 0 and str(train.eval.mode).strip().lower() != "off":
        callbacks.append(
            EvalCheckpointCallback(
                cfg=cfg,
                spec=EvalCheckpointSpec(
                    checkpoint_dir=paths.ckpt_dir,
                    eval_every=int(train.eval.eval_every),
                    eval=train.eval,
                    base_seed=int(run.seed),
                    train_spec=train,
                    verbose=cb_verbose,
                ),
            )
        )
    else:
        logger.info("[eval] disabled (train.eval.mode=off or train.eval.eval_every<=0)")

    cb_list = CallbackList(callbacks)

    if bool(train.rl.enabled) and int(train.rl.total_timesteps) > 0:
        model.learn(
            total_timesteps=int(train.rl.total_timesteps),
            callback=cb_list,
            progress_bar=True,
        )
        model.save(str(paths.ckpt_dir / "final.zip"))
    else:
        logger.info("[rl] skipped (train.rl.enabled is false or total_timesteps <= 0)")

    built.vec_env.close()

    logger.info("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
