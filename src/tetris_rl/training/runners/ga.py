# src/tetris_rl/training/runners/ga.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from planning_rl.ga import GAConfig, GAEvalConfig, HeuristicGA
from planning_rl.ga.types import GAStats
from tetris_rl.config.io import to_plain_dict
from tetris_rl.envs.factory import make_env_from_cfg
from tetris_rl.policies.spec import HeuristicSearch
from tetris_rl.runs.config import RunConfig
from tetris_rl.runs.run_io import make_run_paths, materialize_run_paths
from tetris_rl.runs.run_manifest import write_run_manifest
from tetris_rl.utils.logging import setup_logger


def _parse_ga_config(obj: Any) -> GAConfig:
    if obj is None:
        return GAConfig()
    if isinstance(obj, GAConfig):
        return obj
    if not isinstance(obj, dict):
        raise TypeError("ga config must be a mapping")
    allowed = set(GAConfig.__dataclass_fields__.keys())
    data = {k: v for k, v in obj.items() if k in allowed}
    return GAConfig(**data)


def _parse_eval_config(obj: Any) -> GAEvalConfig:
    if obj is None:
        return GAEvalConfig()
    if isinstance(obj, GAEvalConfig):
        return obj
    if not isinstance(obj, dict):
        raise TypeError("eval config must be a mapping")
    return GAEvalConfig(**obj)


def _log_stats(*, logger: Any, stats: list[GAStats]) -> None:
    if not stats:
        return
    last = stats[-1]
    if last.eval_best_score is None:
        logger.info(
            "[ga] gen=%d best=%.3f mean=%.3f",
            int(last.generation),
            float(last.best_score),
            float(last.mean_score),
        )
        return
    logger.info(
        "[ga] gen=%d best=%.3f mean=%.3f eval_best=%.3f",
        int(last.generation),
        float(last.best_score),
        float(last.mean_score),
        float(last.eval_best_score),
    )


def _with_env_cfg(*, cfg: dict[str, Any], env_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["env"] = dict(env_cfg)
    return out


def run_ga_experiment(cfg: DictConfig) -> int:
    cfg_dict = to_plain_dict(cfg)
    logger = setup_logger(name="tetris_rl.ga", use_rich=True, level=str(cfg_dict.get("log_level", "info")))

    run_cfg = RunConfig.model_validate(cfg_dict.get("run", {}) or {})
    paths = make_run_paths(run_cfg=run_cfg)

    t0 = time.perf_counter()
    materialize_run_paths(paths=paths)
    config_path = paths.run_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(cfg_dict), f=config_path)
    write_run_manifest(run_dir=paths.run_dir, config_path=config_path)
    logger.info(f"[timing] paths+snapshot: {time.perf_counter() - t0:.2f}s")

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

    learn_block = cfg_dict.get("learn", None)
    if not isinstance(learn_block, dict):
        raise TypeError("learn must be a mapping for GA runs")

    algo_block = cfg_dict.get("algo", None)
    if not isinstance(algo_block, dict):
        raise TypeError("algo must be a mapping for GA runs")
    algo_type = str(algo_block.get("type", "")).strip().lower()
    if algo_type != "ga":
        raise ValueError(f"algo.type must be 'ga' for GA runs (got {algo_type!r})")

    ga_cfg = _parse_ga_config(algo_block.get("params", None))
    eval_cfg = _parse_eval_config(learn_block.get("eval", None))

    logger.info(f"[run] dir: {paths.run_dir}")
    logger.info(f"[ga] pop={ga_cfg.population_size} elite={ga_cfg.elite_frac} seed={ga_cfg.seed}")

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

    env_train = make_env_from_cfg(cfg=_with_env_cfg(cfg=cfg_dict, env_cfg=env_train_cfg), seed=int(eval_cfg.seed)).env
    env_eval = make_env_from_cfg(cfg=_with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg), seed=int(eval_cfg.seed)).env

    ga = HeuristicGA(
        cfg=cfg_dict,
        features=features,
        search=search,
        env_train=env_train,
        env_eval=env_eval,
        ga=ga_cfg,
        eval_cfg=eval_cfg,
    )

    generations = int(learn_block.get("generations", 1))
    pop_size = int(ga_cfg.population_size)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        gen_task = progress.add_task("GA generations", total=generations)
        cand_task = progress.add_task("Individuals", total=pop_size)
        current_gen = 0

        def _on_generation(stats: GAStats) -> None:
            label = f"GA gen {int(stats.generation) + 1}"
            progress.update(gen_task, advance=1, description=label)
            progress.update(
                cand_task,
                description=f"Individuals (gen {int(stats.generation) + 1})",
            )

        def _on_candidate(idx: int, score: float) -> None:
            nonlocal current_gen
            if idx == 0:
                current_gen += 1
                progress.update(
                    cand_task,
                    completed=0,
                    total=pop_size,
                    description=f"Individuals (gen {current_gen})",
                )
            _ = score
            progress.update(cand_task, advance=1)

        result = ga.learn(
            generations=generations,
            on_generation=_on_generation,
            on_candidate=_on_candidate,
        )
    _log_stats(logger=logger, stats=result.stats)

    out_spec = paths.run_dir / "best_policy.yaml"
    ga.save_best(path=out_spec, result=result)
    logger.info(f"[ga] best_policy={out_spec}")
    logger.info("[done]")
    return 0


__all__ = ["run_ga_experiment"]
