# src/planning_rl/ga/heuristic.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
import signal
from typing import Any, Callable, Sequence

import numpy as np

from tetris_rl.core.policies.planning_policies.heuristic_policy import HeuristicPlanningPolicy
from tetris_rl.core.policies.spec import HeuristicSearch, HeuristicSpec, save_heuristic_spec
from planning_rl.callbacks import PlanningCallback, wrap_callbacks
from planning_rl.ga.algorithm import GAAlgorithm
from planning_rl.ga.config import GAConfig, GAEvalConfig
from planning_rl.ga.types import GAStats
from tetris_rl.core.envs.factory import make_env_from_cfg


@dataclass(frozen=True)
class HeuristicRunResult:
    spec: HeuristicSpec
    best_score: float
    stats: list[GAStats]

_WORKER_ENV: Any | None = None
_WORKER_POLICY: HeuristicPlanningPolicy | None = None
_WORKER_EVAL_CFG: GAEvalConfig | None = None
_WORKER_SEEDS: list[int] | None = None
_ACTIVE_POOL: Any | None = None


def _eval_policy(
    *,
    policy: HeuristicPlanningPolicy,
    env: Any,
    eval_cfg: GAEvalConfig,
    seeds: Sequence[int],
) -> float:
    total_reward = 0.0
    total_steps = 0
    for seed in seeds:
        _obs, info = env.reset(seed=int(seed))
        _ = info
        steps = 0
        while steps < int(eval_cfg.max_steps):
            action = policy.predict(env=env)
            _obs, reward, terminated, truncated, _info2 = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        total_steps += steps
    if eval_cfg.fitness_metric == "reward_per_step":
        return float(total_reward) / float(max(1, total_steps))
    return float(total_reward)


def _init_worker(
    env_cfg: dict[str, Any],
    features: list[str],
    search_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    seeds: list[int],
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_ENV, _WORKER_POLICY, _WORKER_EVAL_CFG, _WORKER_SEEDS
    _WORKER_EVAL_CFG = GAEvalConfig(**eval_cfg)
    _WORKER_SEEDS = list(seeds)
    search = HeuristicSearch.model_validate(search_cfg)
    _WORKER_POLICY = HeuristicPlanningPolicy(features=features, search=search)
    _WORKER_ENV = make_env_from_cfg(cfg={"env": dict(env_cfg)}, seed=int(_WORKER_EVAL_CFG.seed)).env


def _worker_eval(weights: Sequence[float]) -> float:
    if _WORKER_ENV is None or _WORKER_POLICY is None or _WORKER_EVAL_CFG is None or _WORKER_SEEDS is None:
        raise RuntimeError("worker not initialized")
    _WORKER_POLICY.set_params(weights)
    return _eval_policy(
        policy=_WORKER_POLICY,
        env=_WORKER_ENV,
        eval_cfg=_WORKER_EVAL_CFG,
        seeds=_WORKER_SEEDS,
    )

class HeuristicGA:
    def __init__(
        self,
        *,
        policy: HeuristicPlanningPolicy | None = None,
        cfg: Any | None = None,
        features: Sequence[str] | None = None,
        search: HeuristicSearch | None = None,
        env_train: Any | None = None,
        ga: GAConfig | None = None,
        eval_cfg: GAEvalConfig | None = None,
        workers: int | None = None,
        seed: int | None = None,
    ) -> None:
        if policy is None:
            if cfg is None or features is None:
                raise ValueError("cfg and features are required when policy is not provided")
            policy = HeuristicPlanningPolicy(
                features=list(features),
                search=search,
            )
        self.policy = policy
        self.ga = ga or GAConfig()
        self.workers = int(workers) if workers is not None else 1
        eval_cfg = eval_cfg or GAEvalConfig()
        self._cfg_dict: dict[str, Any] | None = cfg if isinstance(cfg, dict) else None
        self._env_train_cfg: dict[str, Any] | None = None
        if self._cfg_dict is not None:
            env_train_cfg = self._cfg_dict.get("env_train", None) or self._cfg_dict.get("env", None)
            if isinstance(env_train_cfg, dict):
                self._env_train_cfg = dict(env_train_cfg)
        if env_train is None:
            if cfg is None:
                raise ValueError("cfg is required to build env for GA evaluation")
            cfg_env = cfg
            if isinstance(cfg, dict) and "env" not in cfg and "env_train" in cfg:
                cfg_env = dict(cfg)
                cfg_env["env"] = cfg["env_train"]
            env_train = make_env_from_cfg(cfg=cfg_env, seed=int(eval_cfg.seed)).env
        self.env = env_train
        self.algo = GAAlgorithm(
            policy=self.policy,
            env=self.env,
            cfg=self.ga,
            eval_cfg=eval_cfg,
            seed=seed,
        )
        self._best_spec: HeuristicSpec | None = None

    @property
    def best_weights(self) -> list[float]:
        return list(self.algo.best_weights.tolist())

    @property
    def best_score(self) -> float:
        return float(self.algo.best_score)

    def best_agent(self) -> HeuristicPlanningPolicy:
        return self.best_policy()

    def learn(
        self,
        generations: int,
        on_generation: Callable[[GAStats], None] | None = None,
        on_candidate: Callable[[int, float], None] | None = None,
        callback: PlanningCallback | list[PlanningCallback] | None = None,
    ) -> HeuristicRunResult:
        try:
            cb = wrap_callbacks(callback)
            if cb is not None:
                cb.init_callback(self.algo)
                cb.on_start(
                    generations=int(generations),
                    population_size=int(self.algo.population.shape[0]),
                    ga_config=self.ga,
                    eval_config=self.algo.eval_cfg,
                )
            if int(self.workers) <= 1:
                stats = self.algo.learn(
                    generations=int(generations),
                    on_generation=on_generation,
                    on_candidate=on_candidate,
                    callback=cb,
                )
            else:
                if self._env_train_cfg is None:
                    raise ValueError("env_train config required for parallel GA evaluation")
                features = list(self.policy.features)
                search_cfg = self.policy.search.model_dump(mode="json")
                eval_cfg = self.algo.eval_cfg
                seeds = list(self.algo.episode_seeds)
                ctx = get_context("spawn")
                pool = None
                terminated = False
                prev_handler = signal.getsignal(signal.SIGINT)

                def _sigint_handler(_signum, _frame) -> None:
                    if _ACTIVE_POOL is not None:
                        _ACTIVE_POOL.terminate()
                    raise KeyboardInterrupt

                signal.signal(signal.SIGINT, _sigint_handler)
                stats = []
                try:
                    pool = ctx.Pool(
                        processes=int(self.workers),
                        initializer=_init_worker,
                        initargs=(
                            self._env_train_cfg,
                            features,
                            search_cfg,
                            asdict(eval_cfg),
                            seeds,
                        ),
                    )
                    globals()["_ACTIVE_POOL"] = pool
                    for _ in range(int(generations)):
                        if cb is not None:
                            cb.on_event(
                                event="generation_start",
                                generation=int(self.algo.generation),
                                population=self.algo.population,
                            )
                        pop = self.algo.population
                        weights_list = [w.tolist() for w in pop]
                        scores: list[float] = [0.0 for _ in range(len(weights_list))]
                        for idx, score in enumerate(pool.imap(_worker_eval, weights_list)):
                            scores[idx] = float(score)
                            if on_candidate is not None:
                                on_candidate(idx, float(score))
                            if cb is not None:
                                cb.on_event(
                                    event="candidate",
                                    generation=int(self.algo.generation),
                                    candidate_index=int(idx),
                                    score=float(score),
                                    weights=weights_list[int(idx)],
                                )
                        gen_stats = self.algo.tell(scores)
                        stats.append(gen_stats)
                        if on_generation is not None:
                            on_generation(gen_stats)
                        if cb is not None:
                            cb.on_event(
                                event="generation_end",
                                generation=int(gen_stats.generation),
                                stats=gen_stats,
                            )
                except KeyboardInterrupt:
                    terminated = True
                    if pool is not None:
                        pool.terminate()
                        pool.join()
                    raise
                finally:
                    globals()["_ACTIVE_POOL"] = None
                    signal.signal(signal.SIGINT, prev_handler)
                    if pool is not None and not terminated:
                        pool.close()
                        pool.join()
            spec = self.policy.build_spec(self.best_weights)
            self._best_spec = spec
            if cb is not None:
                cb.on_end(
                    stats=list(stats),
                    best_score=float(self.algo.best_score),
                    best_weights=self.algo.best_weights.tolist(),
                )
            return HeuristicRunResult(
                spec=spec,
                best_score=float(self.algo.best_score),
                stats=stats,
            )
        finally:
            self.env.close()

    def save_best(self, *, path: Path, result: HeuristicRunResult) -> Path:
        return save_heuristic_spec(path, result.spec)

    def best_policy(self) -> HeuristicPlanningPolicy:
        if self._best_spec is None:
            raise RuntimeError("no best_spec available; call learn() first")
        return HeuristicPlanningPolicy.from_spec(self._best_spec)

    def predict(self, *, env: Any, game: Any) -> Any:
        policy = self.best_policy()
        _ = game
        return policy.predict(env=env)


__all__ = [
    "HeuristicGA",
    "HeuristicRunResult",
]
