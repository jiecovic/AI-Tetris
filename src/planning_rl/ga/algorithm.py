# src/planning_rl/ga/algorithm.py
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from planning_rl.algorithms import PlanningAlgorithm
from planning_rl.ga.config import GAConfig, GAEvalConfig
from planning_rl.ga.types import GAStats
from planning_rl.policies import VectorParamPolicy
from planning_rl.utils.seed import seed32_from

def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    if weights.ndim == 1:
        norm = float(np.linalg.norm(weights))
        if norm <= 0.0:
            return weights
        return weights / norm
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return weights / norms


def _init_population(*, cfg: GAConfig, num_params: int, rng: np.random.Generator) -> np.ndarray:
    if num_params <= 0:
        raise ValueError("num_params must be >= 1")
    lo, hi = cfg.weight_init_range
    if lo >= hi:
        raise ValueError("weight_init_range must be (lo, hi) with lo < hi")
    pop = rng.uniform(lo, hi, size=(cfg.population_size, num_params)).astype(np.float64)
    if cfg.normalize_weights:
        pop = _normalize_weights(pop)
    return pop


def _mutate(weights: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    out = weights.copy()
    if cfg.mutation_kind == "gaussian":
        mask = rng.random(out.shape) < float(cfg.mutation_rate)
        if mask.any():
            noise = rng.normal(loc=0.0, scale=float(cfg.mutation_sigma), size=out.shape)
            out[mask] += noise[mask]
    elif cfg.mutation_kind == "single_component":
        if rng.random() < float(cfg.mutation_rate):
            idx = int(rng.integers(0, out.shape[0]))
            delta = float(rng.uniform(-float(cfg.mutation_delta), float(cfg.mutation_delta)))
            out[idx] += delta
    else:
        raise ValueError(f"unsupported mutation_kind: {cfg.mutation_kind}")
    if cfg.weight_clip is not None:
        lo, hi = cfg.weight_clip
        out = np.clip(out, lo, hi)
    if cfg.normalize_weights:
        out = _normalize_weights(out)
    return out


def _crossover(
    a: np.ndarray,
    b: np.ndarray,
    score_a: float | None,
    score_b: float | None,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if rng.random() >= float(cfg.crossover_rate):
        return a.copy()
    if cfg.crossover_kind == "uniform_mask":
        mask = rng.random(a.shape) < 0.5
        return np.where(mask, a, b)
    if cfg.crossover_kind == "weighted_avg":
        if score_a is None or score_b is None:
            raise ValueError("weighted_avg crossover requires parent scores")
        wa = max(0.0, float(score_a))
        wb = max(0.0, float(score_b))
        if wa + wb <= 0.0:
            return 0.5 * (a + b)
        return (wa * a + wb * b) / (wa + wb)
    raise ValueError(f"unsupported crossover_kind: {cfg.crossover_kind}")


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
    order = np.argsort(np.asarray(scores))[::-1]
    scores_arr = np.asarray(scores, dtype=np.float64)

    next_pop = np.empty_like(population)
    if cfg.selection == "elite_pool":
        elite_count = max(1, int(pop_size * float(cfg.elite_frac)))
        parent_pool_count = max(2, int(pop_size * float(cfg.parent_pool())))
        elites = population[order[:elite_count]]
        next_pop[:elite_count] = elites

        parent_pool = population[order[:parent_pool_count]]
        parent_scores = scores_arr[order[:parent_pool_count]]
        for i in range(elite_count, pop_size):
            ia = int(rng.integers(0, parent_pool.shape[0]))
            ib = int(rng.integers(0, parent_pool.shape[0]))
            child = _crossover(
                parent_pool[ia],
                parent_pool[ib],
                float(parent_scores[ia]),
                float(parent_scores[ib]),
                cfg,
                rng,
            )
            child = _mutate(child, cfg, rng)
            next_pop[i] = child
    elif cfg.selection == "tournament":
        offspring_count = max(1, int(pop_size * float(cfg.offspring_frac)))
        keep_count = pop_size - offspring_count
        next_pop[:keep_count] = population[order[:keep_count]]
        pool_size = max(2, int(pop_size * float(cfg.tournament_frac)))
        pool_size = min(pool_size, pop_size)
        winners_count = max(2, int(cfg.tournament_winners))
        winners_count = min(winners_count, pool_size)

        for i in range(offspring_count):
            pool_idx = rng.choice(pop_size, size=pool_size, replace=False)
            pool_scores = scores_arr[pool_idx]
            pool_order = np.argsort(pool_scores)[::-1]
            winners = pool_idx[pool_order[:winners_count]]
            if winners_count == 2:
                ia, ib = int(winners[0]), int(winners[1])
            else:
                ia, ib = rng.choice(winners, size=2, replace=False)
                ia, ib = int(ia), int(ib)

            child = _crossover(
                population[ia],
                population[ib],
                float(scores_arr[ia]),
                float(scores_arr[ib]),
                cfg,
                rng,
            )
            child = _mutate(child, cfg, rng)
            next_pop[keep_count + i] = child
    else:
        raise ValueError(f"unsupported selection strategy: {cfg.selection}")

    best_idx = int(order[0])
    best_score = float(scores[best_idx])
    mean_score = float(np.mean(scores))
    stats = GAStats(
        generation=int(generation),
        best_score=best_score,
        mean_score=mean_score,
        best_index=best_idx,
        best_weights=population[best_idx].tolist(),
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
        if cfg.fitness_metric == "reward_per_step":
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
