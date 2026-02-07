# src/planning_rl/ga/algorithm.py
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from planning_rl.algorithms import PlanningAlgorithm
from planning_rl.ga.config import GAConfig, GAEvalConfig
from planning_rl.ga.types import GAStats
from planning_rl.policies import VectorParamPolicy
from planning_rl.utils.seed import seed32_from

def _init_population(*, cfg: GAConfig, num_params: int, rng: np.random.Generator) -> np.ndarray:
    if num_params <= 0:
        raise ValueError("num_params must be >= 1")
    lo, hi = cfg.weight_init_range
    if lo >= hi:
        raise ValueError("weight_init_range must be (lo, hi) with lo < hi")
    pop = rng.uniform(lo, hi, size=(cfg.population_size, num_params)).astype(np.float64)
    return pop


def _mutate(weights: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    out = weights.copy()
    mask = rng.random(out.shape) < float(cfg.mutation_rate)
    if mask.any():
        noise = rng.normal(loc=0.0, scale=float(cfg.mutation_sigma), size=out.shape)
        out[mask] += noise[mask]
    if cfg.weight_clip is not None:
        lo, hi = cfg.weight_clip
        out = np.clip(out, lo, hi)
    return out


def _crossover(
    a: np.ndarray,
    b: np.ndarray,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if rng.random() >= float(cfg.crossover_rate):
        return a.copy()
    mask = rng.random(a.shape) < 0.5
    child = np.where(mask, a, b)
    return child


def _episode_seeds(*, base_seed: int, episodes: int) -> list[int]:
    return [int(seed32_from(base_seed=int(base_seed), stream_id=int(i))) for i in range(episodes)]


def _evolve_population(
    *,
    cfg: GAConfig,
    population: np.ndarray,
    scores: Sequence[float],
    rng: np.random.Generator,
    generation: int,
    eval_best_score: float | None = None,
) -> tuple[np.ndarray, GAStats]:
    if population.shape[0] != len(scores):
        raise ValueError("population and scores length mismatch")

    pop_size = population.shape[0]
    elite_count = max(1, int(pop_size * float(cfg.elite_frac)))
    parent_pool_count = max(2, int(pop_size * float(cfg.parent_pool())))

    order = np.argsort(np.asarray(scores))[::-1]
    elites = population[order[:elite_count]]

    next_pop = np.empty_like(population)
    next_pop[:elite_count] = elites

    parent_pool = population[order[:parent_pool_count]]
    for i in range(elite_count, pop_size):
        ia = int(rng.integers(0, parent_pool.shape[0]))
        ib = int(rng.integers(0, parent_pool.shape[0]))
        child = _crossover(parent_pool[ia], parent_pool[ib], cfg, rng)
        child = _mutate(child, cfg, rng)
        next_pop[i] = child

    best_idx = int(order[0])
    best_score = float(scores[best_idx])
    mean_score = float(np.mean(scores))
    stats = GAStats(
        generation=int(generation),
        best_score=best_score,
        mean_score=mean_score,
        best_index=best_idx,
        eval_best_score=eval_best_score,
    )
    return next_pop, stats


class GAAlgorithm(PlanningAlgorithm):
    def __init__(
        self,
        *,
        policy: VectorParamPolicy,
        env: Any,
        eval_env: Any | None = None,
        cfg: GAConfig,
        eval_cfg: GAEvalConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.env = env
        self.eval_env = eval_env
        self.cfg = cfg
        self.eval_cfg = eval_cfg or GAEvalConfig()
        self.episode_seeds = _episode_seeds(
            base_seed=int(self.eval_cfg.seed),
            episodes=int(self.eval_cfg.episodes),
        )
        self.rng = np.random.default_rng(int(seed if seed is not None else cfg.seed))
        self.population = _init_population(cfg=cfg, num_params=policy.num_params, rng=self.rng)
        self.generation = 0
        self.stats: list[GAStats] = []
        self.best_weights = self.population[0].copy()
        self.best_score = float("-inf")

    def ask(self) -> np.ndarray:
        return self.population

    def predict(self, *, env: Any | None = None) -> Any:
        return super().predict(env=self.env if env is None else env)

    def tell(self, scores: Sequence[float], eval_best_score: float | None = None) -> GAStats:
        if len(scores) != self.population.shape[0]:
            raise ValueError("scores length must match population size")

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_weights = self.population[best_idx].copy()

        self.population, stats = _evolve_population(
            cfg=self.cfg,
            population=self.population,
            scores=scores,
            rng=self.rng,
            generation=self.generation,
            eval_best_score=eval_best_score,
        )
        self.stats.append(stats)
        self.generation += 1
        return stats

    def evaluate_weights(
        self,
        *,
        weights: Sequence[float],
        env: Any | None = None,
        eval_cfg: GAEvalConfig | None = None,
    ) -> float:
        cfg = eval_cfg or self.eval_cfg
        env_eval = self.env if env is None else env
        if eval_cfg is None:
            seeds = self.episode_seeds
        else:
            seeds = _episode_seeds(base_seed=int(cfg.seed), episodes=int(cfg.episodes))
        self.policy.set_params(weights)
        total_reward = 0.0
        total_steps = 0
        for seed in seeds:
            _obs, info = env_eval.reset(seed=int(seed))
            _ = info
            steps = 0
            while steps < int(cfg.max_steps):
                action = self.policy.predict(env=env_eval)
                _obs, reward, terminated, truncated, _info2 = env_eval.step(action)
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            total_steps += steps
        if cfg.return_metric == "reward_per_step":
            return float(total_reward) / float(max(1, total_steps))
        return float(total_reward)

    def evaluate_population(
        self,
        *,
        on_candidate: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        scores: list[float] = []
        for i in range(self.population.shape[0]):
            weights = self.population[i].tolist()
            score = float(self.evaluate_weights(weights=weights, env=self.env))
            scores.append(score)
            if on_candidate is not None:
                on_candidate(i, score)
        return scores

    def learn(
        self,
        *,
        generations: int,
        on_generation: Callable[[GAStats], None] | None = None,
        on_candidate: Callable[[int, float], None] | None = None,
    ) -> list[GAStats]:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        for _ in range(int(generations)):
            scores = self.evaluate_population(on_candidate=on_candidate)
            eval_best_score = None
            if self.eval_env is not None:
                best_idx = int(np.argmax(scores))
                best_weights = self.population[best_idx].tolist()
                eval_best_score = float(self.evaluate_weights(weights=best_weights, env=self.eval_env))
            stats = self.tell(scores, eval_best_score=eval_best_score)
            if on_generation is not None:
                on_generation(stats)
        return list(self.stats)


__all__ = ["GAAlgorithm"]
