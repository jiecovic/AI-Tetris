# src/tetris_rl/core/training/runners/ga.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.rich import FractionColumn, RateColumn

from planning_rl.ga import GAConfig, GAEvalConfig, HeuristicGA
from planning_rl.ga.types import GAStats
from tetris_rl.core.callbacks import EvalCallback, LatestCallback, PlanningCallbackAdapter
from tetris_rl.core.config.io import to_plain_dict
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.policies.spec import HeuristicSearch, save_heuristic_spec
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.runs.run_io import make_run_paths, materialize_run_paths
from tetris_rl.core.runs.run_manifest import write_run_manifest
from tetris_rl.core.training.config import CallbacksConfig, EvalConfig
from tetris_rl.core.training.evaluation import evaluate_planning_policy
from tetris_rl.core.training.evaluation.eval_checkpoint_core import EvalCheckpointCoreSpec
from tetris_rl.core.training.evaluation.latest_checkpoint_core import LatestCheckpointCoreSpec
from tetris_rl.core.utils.logging import setup_logger


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
    fitness_cfg = _parse_eval_config(learn_block.get("eval", None))
    callbacks_cfg = CallbacksConfig.model_validate(cfg_dict.get("callbacks", {}) or {})
    eval_cfg = EvalConfig.model_validate(cfg_dict.get("eval", {}) or {})

    logger.info(f"[run] dir: {paths.run_dir}")
    logger.info(
        "[ga] pop=%d elite=%.2f selection=%s crossover=%s mutation=%s normalize=%s workers=%d",
        int(ga_cfg.population_size),
        float(ga_cfg.elite_frac),
        str(ga_cfg.selection),
        str(ga_cfg.crossover_kind),
        str(ga_cfg.mutation_kind),
        bool(ga_cfg.normalize_weights),
        int(run_cfg.workers),
    )
    logger.info(
        "[ga] fitness episodes=%d max_steps=%d fitness=%s seed=%d",
        int(fitness_cfg.episodes),
        int(fitness_cfg.max_steps),
        str(fitness_cfg.fitness_metric),
        int(fitness_cfg.seed),
    )
    logger.info(
        "[ga] eval every=%d steps=%d seed_offset=%d",
        int(callbacks_cfg.eval.every),
        int(eval_cfg.steps),
        int(eval_cfg.seed_offset),
    )
    logger.info(
        "[ga] callbacks latest_enabled=%s latest_every=%d eval_enabled=%s eval_every=%d",
        bool(callbacks_cfg.latest.enabled),
        int(callbacks_cfg.latest.every),
        bool(callbacks_cfg.eval.enabled),
        int(callbacks_cfg.eval.every),
    )
    logger.info(
        "[ga] search plies=%d beam_width=%s beam_from_depth=%d",
        int(search.plies),
        str(search.beam_width),
        int(search.beam_from_depth),
    )
    logger.info("[ga] features=%s", ",".join(str(f) for f in features))
    if ga_cfg.selection == "tournament":
        logger.info(
            "[ga] tournament frac=%.2f winners=%d offspring=%.2f",
            float(ga_cfg.tournament_frac),
            int(ga_cfg.tournament_winners),
            float(ga_cfg.offspring_frac),
        )
    if ga_cfg.selection == "elite_pool":
        logger.info(
            "[ga] elite_pool parent_pool=%.2f",
            float(ga_cfg.parent_pool()),
        )
    if ga_cfg.crossover_kind == "weighted_avg":
        logger.info("[ga] crossover=weighted_avg rate=%.2f", float(ga_cfg.crossover_rate))
    if ga_cfg.crossover_kind == "uniform_mask":
        logger.info("[ga] crossover=uniform_mask rate=%.2f", float(ga_cfg.crossover_rate))
    if ga_cfg.mutation_kind == "single_component":
        logger.info(
            "[ga] mutation=single_component rate=%.2f delta=%.2f",
            float(ga_cfg.mutation_rate),
            float(ga_cfg.mutation_delta),
        )
    if ga_cfg.mutation_kind == "gaussian":
        logger.info(
            "[ga] mutation=gaussian rate=%.2f sigma=%.2f",
            float(ga_cfg.mutation_rate),
            float(ga_cfg.mutation_sigma),
        )

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

    env_train = make_env_from_cfg(
        cfg=_with_env_cfg(cfg=cfg_dict, env_cfg=env_train_cfg),
        seed=int(fitness_cfg.seed),
    ).env

    eval_enabled = bool(callbacks_cfg.eval.enabled) and int(callbacks_cfg.eval.every) > 0
    env_eval = None
    if eval_enabled:
        env_eval = make_env_from_cfg(
            cfg=_with_env_cfg(cfg=cfg_dict, env_cfg=env_eval_cfg),
            seed=int(fitness_cfg.seed),
        ).env

    ga = HeuristicGA(
        cfg=cfg_dict,
        features=features,
        search=search,
        env_train=env_train,
        ga=ga_cfg,
        eval_cfg=fitness_cfg,
        workers=int(run_cfg.workers),
    )

    generations = int(learn_block.get("generations", 1))
    pop_size = int(ga_cfg.population_size)

    callback_items: list[PlanningCallbackAdapter] = []
    latest_every = int(callbacks_cfg.latest.every)
    if bool(callbacks_cfg.latest.enabled) and latest_every > 0:
        core_cb = LatestCallback(
            spec=LatestCheckpointCoreSpec(
                checkpoint_dir=paths.ckpt_dir,
                latest_every=latest_every,
                verbose=0,
            ),
            event="generation_end",
            progress_key="generation",
            progress_offset=1,
        )
        callback_items.append(PlanningCallbackAdapter(core_cb))

    if eval_enabled:
        eval_seed_base = int(run_cfg.seed) + int(eval_cfg.seed_offset)

        def _eval_fn(model: Any, _t: int, on_episode, on_step) -> dict[str, Any]:
            if env_eval is None:
                raise RuntimeError("eval env not initialized")
            policy = getattr(model, "policy", None)
            if policy is None:
                raise RuntimeError("GA eval requires algo.policy")
            weights = getattr(model, "best_weights", None)
            if weights is not None:
                try:
                    policy.set_params(weights.tolist())
                except Exception:
                    policy.set_params(list(weights))
            return evaluate_planning_policy(
                policy=policy,
                env=env_eval,
                eval_steps=int(eval_cfg.steps),
                seed_base=int(eval_seed_base),
                deterministic=bool(eval_cfg.deterministic),
                on_episode=on_episode,
                on_step=on_step,
            )

        core_cb = EvalCallback(
            spec=EvalCheckpointCoreSpec(
                checkpoint_dir=paths.ckpt_dir,
                eval_every=int(callbacks_cfg.eval.every),
                run_cfg=run_cfg,
                eval=eval_cfg,
                base_seed=int(run_cfg.seed),
                progress_unit="generations",
                verbose=1,
            ),
            cfg=cfg_dict,
            event="generation_end",
            progress_key="generation",
            progress_offset=1,
            phase="ga",
            eval_fn=_eval_fn,
        )
        callback_items.append(PlanningCallbackAdapter(core_cb))
    else:
        logger.info("[eval] disabled (callbacks.eval disabled)")

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
            gen_task = progress.add_task(
                "GA generations",
                total=generations,
                tail="gen_best=- gen_w=-",
            )
            cand_task = progress.add_task(
                "Individuals (gen 1)",
                total=pop_size,
                tail="best_ind=- best_w=-",
            )
            current_gen = 0
            best_ind: float | None = None
            best_ind_weights: list[float] | None = None
            gen_best: float | None = None
            gen_best_weights: list[float] | None = None

            def _fmt(v: float | None) -> str:
                return "-" if v is None else f"{float(v):.3f}"

            def _fmt_weights(weights: list[float] | None) -> str:
                if not weights:
                    return "-"
                items = [f"{float(w):.3f}" for w in weights]
                return "[" + ",".join(items) + "]"

            best_path = paths.run_dir / "best_policy.yaml"
            intermediate_path = paths.run_dir / "intermediate_best_policy.yaml"

            def _save_best(path: Path) -> None:
                spec = ga.policy.build_spec(ga.best_weights)
                save_heuristic_spec(path, spec)

            def _on_generation(stats: GAStats) -> None:
                nonlocal gen_best
                nonlocal gen_best_weights
                label = f"GA gen {int(stats.generation) + 1}"
                progress.update(gen_task, advance=1, description=label)
                gen_best = float(stats.best_score)
                gen_best_weights = stats.best_weights
                progress.update(
                    cand_task,
                    description=f"Individuals (gen {int(stats.generation) + 1})",
                )
                progress.update(
                    gen_task,
                    tail=f"gen_best={_fmt(gen_best)} gen_w={_fmt_weights(gen_best_weights)}",
                )
                progress.update(
                    cand_task,
                    tail=f"best_ind={_fmt(best_ind)} best_w={_fmt_weights(best_ind_weights)}",
                )
                _save_best(intermediate_path)

            def _on_candidate(idx: int, score: float) -> None:
                nonlocal current_gen
                nonlocal best_ind
                nonlocal best_ind_weights
                if idx == 0:
                    current_gen += 1
                    best_ind = None
                    best_ind_weights = None
                    progress.update(
                        cand_task,
                        completed=0,
                        total=pop_size,
                        description=f"Individuals (gen {current_gen})",
                        tail=f"best_ind={_fmt(best_ind)} best_w={_fmt_weights(best_ind_weights)}",
                    )
                if best_ind is None or float(score) > float(best_ind):
                    best_ind = float(score)
                    best_ind_weights = [float(w) for w in ga.algo.population[int(idx)].tolist()]
                    progress.update(
                        cand_task,
                        tail=f"best_ind={_fmt(best_ind)} best_w={_fmt_weights(best_ind_weights)}",
                    )
                progress.update(cand_task, advance=1)

            result = ga.learn(
                generations=generations,
                on_generation=_on_generation,
                on_candidate=_on_candidate,
                callback=callback_items or None,
            )
    finally:
        if env_eval is not None:
            env_eval.close()
    _log_stats(logger=logger, stats=result.stats)

    ga.save_best(path=best_path, result=result)
    logger.info(f"[ga] best_policy={best_path}")
    if intermediate_path.is_file():
        logger.info(f"[ga] intermediate_best_policy={intermediate_path}")
    logger.info("[done]")
    return 0


__all__ = ["run_ga_experiment"]
