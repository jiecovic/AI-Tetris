# src/tetris_rl/core/training/runners/td.py
from __future__ import annotations

import time
from typing import Any
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.rich import FractionColumn, RateColumn

from planning_rl.callbacks import PlanningCallback
from planning_rl.td import LinearValueModel, TDAlgorithm, TDConfig
from tetris_rl.core.callbacks import EvalCallback, LatestCallback, PlanningCallbackAdapter
from tetris_rl.core.config.io import to_plain_dict
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.policies.spec import HeuristicSearch
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.runs.run_io import make_run_paths, materialize_run_paths
from tetris_rl.core.runs.run_manifest import write_run_manifest
from tetris_rl.core.training.config import CallbacksConfig
from tetris_rl.core.training.evaluation import (
    evaluate_planning_policy,
    evaluate_planning_policy_parallel,
)
from tetris_rl.core.training.evaluation.eval_checkpoint_core import EvalCheckpointCoreSpec
from tetris_rl.core.training.evaluation.latest_checkpoint_core import LatestCheckpointCoreSpec
from tetris_rl.core.training.reporting import log_env_reward_summary
from tetris_rl.core.training.tb_logger import maybe_tb_logger
from tetris_rl.core.utils.logging import setup_logger
from planning_rl.utils.seed import seed32_from
from tetris_rl.core.policies.planning_policies.td_value_policy import TDValuePlanningPolicy
from tetris_rl.core.utils.seed import seed_all


def _parse_td_config(obj: Any, *, seed_default: int) -> TDConfig:
    if obj is None:
        return TDConfig(seed=int(seed_default))
    if isinstance(obj, TDConfig):
        return obj
    if not isinstance(obj, dict):
        raise TypeError("TD params must be a mapping for TD runs")
    allowed = set(TDConfig.__dataclass_fields__.keys())
    data = {k: v for k, v in obj.items() if k in allowed}
    if "seed" not in data:
        data["seed"] = int(seed_default)
    return TDConfig(**data)


def _with_env_cfg(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    max_steps_per_episode: int | None = None,
) -> dict[str, Any]:
    out = dict(cfg)
    env_out = dict(env_cfg)
    if max_steps_per_episode is not None:
        params = env_out.get("params", {}) or {}
        if not isinstance(params, dict):
            params = {}
        params = dict(params)
        params["max_steps"] = int(max_steps_per_episode)
        env_out["params"] = params
    out["env"] = env_out
    return out


def _device_from_run(run_cfg: RunConfig) -> torch.device:
    dev = str(getattr(run_cfg, "device", "auto")).strip().lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def run_td_experiment(cfg: DictConfig) -> int:
    cfg_dict = to_plain_dict(cfg)
    logger = setup_logger(name="tetris_rl.td", use_rich=True, level=str(cfg_dict.get("log_level", "info")))

    run_cfg = RunConfig.model_validate(cfg_dict.get("run", {}) or {})
    seed_all(int(run_cfg.seed))
    paths = make_run_paths(run_cfg=run_cfg)

    t0 = time.perf_counter()
    materialize_run_paths(paths=paths)
    config_path = paths.run_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(cfg_dict), f=config_path)
    write_run_manifest(run_dir=paths.run_dir, config_path=config_path)
    logger.info(f"[timing] paths+snapshot: {time.perf_counter() - t0:.2f}s")
    tb_logger = maybe_tb_logger(paths.tb_dir)

    policy_cfg = cfg_dict.get("policy", {}) or {}
    if not isinstance(policy_cfg, dict):
        raise TypeError("policy must be a mapping")

    features = policy_cfg.get("features", []) or []
    if not isinstance(features, list) or not features:
        raise ValueError("policy.features must be a non-empty list")

    search_cfg = policy_cfg.get("search", {}) or {}
    if not isinstance(search_cfg, dict):
        raise TypeError("policy.search must be a mapping")
    search = HeuristicSearch.model_validate(search_cfg)

    algo_block = cfg_dict.get("algo", None)
    if not isinstance(algo_block, dict):
        raise TypeError("algo must be a mapping for TD runs")
    algo_type = str(algo_block.get("type", "")).strip().lower()
    if algo_type != "td":
        raise ValueError(f"algo.type must be 'td' for TD runs (got {algo_type!r})")

    learn_block = cfg_dict.get("learn", None)
    if learn_block is None:
        learn_block = {}
    if not isinstance(learn_block, dict):
        raise TypeError("learn must be a mapping for TD runs")

    algo_params = algo_block.get("params", None)
    if algo_params is None:
        algo_params = {}
    if not isinstance(algo_params, dict):
        raise TypeError("algo.params must be a mapping for TD runs")

    td_cfg = _parse_td_config({**learn_block, **algo_params}, seed_default=int(run_cfg.seed))
    callbacks_cfg = CallbacksConfig.model_validate(cfg_dict.get("callbacks", {}) or {})
    eval_cfg = callbacks_cfg.eval_checkpoint

    env_train_cfg = cfg_dict.get("env_train", None)
    env_eval_cfg = cfg_dict.get("env_eval", None)
    env_fallback = cfg_dict.get("env", None)

    if env_train_cfg is None:
        env_train_cfg = env_fallback
    if env_eval_cfg is None:
        env_eval_cfg = env_train_cfg

    if not isinstance(env_train_cfg, dict):
        raise TypeError("env_train must be a mapping (or define env as fallback)")
    if not isinstance(env_eval_cfg, dict):
        raise TypeError("env_eval must be a mapping (or define env as fallback)")

    n_envs = max(1, int(td_cfg.n_envs))
    envs: list[Any] = []
    for i in range(n_envs):
        seed_i = int(seed32_from(base_seed=int(td_cfg.seed), stream_id=int(0x7D00 + i)))
        built_env = make_env_from_cfg(
                cfg=_with_env_cfg(
                    cfg=cfg_dict,
                    env_cfg=env_train_cfg,
                    max_steps_per_episode=td_cfg.max_steps_per_episode,
                ),
                seed=seed_i,
            )
        if i == 0:
            log_env_reward_summary(logger=logger, label="train", built_env=built_env, env_cfg=env_train_cfg)
        envs.append(built_env.env)

    device = _device_from_run(run_cfg)
    value_model = LinearValueModel(
        num_features=int(len(features)),
        weight_norm=str(getattr(td_cfg, "weight_norm", "none")),
        weight_scale=float(getattr(td_cfg, "weight_scale", 1.0)),
        learn_scale=bool(getattr(td_cfg, "learn_scale", True)),
        weight_norm_eps=float(getattr(td_cfg, "weight_norm_eps", 1e-8)),
    ).to(device=device)
    if float(td_cfg.weight_init_std) > 0.0:
        with torch.no_grad():
            value_model.weights.normal_(mean=0.0, std=float(td_cfg.weight_init_std))

    policy = TDValuePlanningPolicy(features=list(features), search=search, value_model=value_model)
    algo = TDAlgorithm(
        policy=policy,
        envs=envs,
        features=list(features),
        cfg=td_cfg,
        device=device,
    )
    algo.policy.sync_from_model()
    algo.optimizer = torch.optim.Adam(value_model.parameters(), lr=float(td_cfg.learning_rate))

    logger.info(
        "[td] steps=%d rollout=%d n_envs=%d gamma=%.3f gae=%.3f batch=%d epochs=%d",
        int(td_cfg.total_timesteps),
        int(td_cfg.rollout_steps),
        int(td_cfg.n_envs),
        float(td_cfg.gamma),
        float(td_cfg.gae_lambda),
        int(td_cfg.batch_size),
        int(td_cfg.n_epochs),
    )
    logger.info(
        "[td] lr=%.6f grad_clip=%.3f clip_range_vf=%.3f weight_init_std=%.3f seed=%d",
        float(td_cfg.learning_rate),
        float(td_cfg.grad_clip),
        float(td_cfg.clip_range_vf),
        float(td_cfg.weight_init_std),
        int(td_cfg.seed),
    )
    logger.info(
        "[td] weight_norm=%s scale=%.3f learn_scale=%s eps=%.1e",
        str(getattr(td_cfg, "weight_norm", "none")),
        float(getattr(td_cfg, "weight_scale", 1.0)),
        bool(getattr(td_cfg, "learn_scale", True)),
        float(getattr(td_cfg, "weight_norm_eps", 1e-8)),
    )
    logger.info(
        "[td] search plies=%d beam_width=%s beam_from_depth=%d",
        int(search.plies),
        str(search.beam_width),
        int(search.beam_from_depth),
    )
    logger.info("[td] features=%s", ",".join(str(f) for f in features))

    callback_items: list[PlanningCallbackAdapter] = []
    latest_every = int(callbacks_cfg.latest.every)
    if bool(callbacks_cfg.latest.enabled) and latest_every > 0:
        core_cb = LatestCallback(
            spec=LatestCheckpointCoreSpec(
                checkpoint_dir=paths.ckpt_dir,
                latest_every=latest_every,
                verbose=0,
            ),
            event="step",
            progress_key="num_timesteps",
            progress_offset=0,
        )
        callback_items.append(PlanningCallbackAdapter(core_cb))

    eval_enabled = bool(callbacks_cfg.eval_checkpoint.enabled) and int(callbacks_cfg.eval_checkpoint.every) > 0
    eval_workers = int(eval_cfg.workers)
    env_eval = None
    if eval_enabled and int(eval_workers) <= 1:
        built_eval = make_env_from_cfg(
            cfg=_with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg),
            seed=int(td_cfg.seed),
        )
        env_eval = built_eval.env
        log_env_reward_summary(logger=logger, label="eval", built_env=built_eval, env_cfg=env_eval_cfg)

    last_eval_weights: list[float] | None = None
    if eval_enabled:
        eval_seed_base = int(run_cfg.seed) + int(eval_cfg.seed_offset)

        def _eval_fn(model: Any, _t: int, on_episode, on_step) -> dict[str, Any]:
            nonlocal last_eval_weights
            policy = getattr(model, "policy", None)
            if policy is None:
                raise RuntimeError("TD eval requires algo.policy")
            model.policy.sync_from_model()
            try:
                last_eval_weights = [float(w) for w in policy.get_params()]
            except Exception:
                last_eval_weights = None
            if int(eval_workers) > 1:
                spec = policy.build_spec(policy.get_params())
                return evaluate_planning_policy_parallel(
                    spec=spec,
                    env_cfg=env_eval_cfg,
                    eval_episodes=int(eval_cfg.episodes),
                    min_steps=int(eval_cfg.min_steps),
                    max_steps_per_episode=eval_cfg.max_steps_per_episode,
                    seed_base=int(eval_seed_base),
                    deterministic=bool(eval_cfg.deterministic),
                    workers=int(eval_workers),
                    on_episode=on_episode,
                    on_step=on_step,
                )
            if env_eval is None:
                raise RuntimeError("eval env not initialized")
            return evaluate_planning_policy(
                policy=policy,
                env=env_eval,
                eval_episodes=int(eval_cfg.episodes),
                min_steps=int(eval_cfg.min_steps),
                max_steps_per_episode=eval_cfg.max_steps_per_episode,
                seed_base=int(eval_seed_base),
                deterministic=bool(eval_cfg.deterministic),
                on_episode=on_episode,
                on_step=on_step,
            )

        def _extra_metrics() -> dict[str, Any]:
            if last_eval_weights is None:
                return {}
            return {"policy/weights": list(last_eval_weights)}

        core_cb = EvalCallback(
            spec=EvalCheckpointCoreSpec(
                checkpoint_dir=paths.ckpt_dir,
                eval_every=int(callbacks_cfg.eval_checkpoint.every),
                run_cfg=run_cfg,
                eval=eval_cfg,
                base_seed=int(run_cfg.seed),
                progress_unit="steps",
                verbose=1,
            ),
            cfg=cfg_dict,
            event="step",
            progress_key="num_timesteps",
            progress_offset=0,
            phase="td",
            emit=logger.info,
            eval_fn=_eval_fn,
            extra_metrics_fn=_extra_metrics,
            log_scalar=(tb_logger.log_scalar if tb_logger is not None else None),
        )
        callback_items.append(PlanningCallbackAdapter(core_cb))
    else:
        logger.info("[eval] disabled (callbacks.eval_checkpoint disabled)")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>4.0f}%"),
        BarColumn(bar_width=None),
        FractionColumn(),
        TextColumn("["),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn(","),
        RateColumn(unit="it"),
        TextColumn("]"),
        TextColumn("{task.fields[tail]}"),
    )

    try:
        with progress:
            task = progress.add_task(
                "TD steps",
                total=int(td_cfg.total_timesteps),
                tail="loss=-",
            )

            class _ProgressCallback(PlanningCallback):
                def __init__(self, *, progress: Progress, task_id: int) -> None:
                    super().__init__()
                    self._progress = progress
                    self._task_id = task_id
                    self._best_ret: float | None = None

                @staticmethod
                def _fmt(v: float | None) -> str:
                    return "-" if v is None else f"{float(v):.3f}"

                def on_event(self, *, event: str, **kwargs: Any) -> None:
                    if event != "step":
                        return
                    step = kwargs.get("num_timesteps")
                    if step is None:
                        return
                    loss = None
                    if algo.stats:
                        last_stats = algo.stats[-1]
                        loss = last_stats.get("loss")
                        last_ret = last_stats.get("ep_ret_mean")
                        if last_ret is not None:
                            last_ret = float(last_ret)
                            if self._best_ret is None or last_ret > self._best_ret:
                                self._best_ret = last_ret
                    loss_str = "loss=-" if loss is None else f"loss={float(loss):.5f}"
                    best_str = f"best_ret={self._fmt(self._best_ret)}"
                    tail = f"{loss_str} {best_str}"
                    self._progress.update(self._task_id, completed=int(step), tail=tail)

            cb_items = list(callback_items)
            cb_items.append(_ProgressCallback(progress=progress, task_id=task))

            algo.learn(
                callback=cb_items,
                logger=tb_logger,
            )
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        if env_eval is not None:
            env_eval.close()
        if tb_logger is not None:
            tb_logger.flush()
            tb_logger.close()

    logger.info("[done]")
    return 0


__all__ = ["run_td_experiment"]
