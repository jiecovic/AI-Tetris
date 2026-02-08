# src/planning_rl/ga/operators.py
from __future__ import annotations

from typing import Sequence

import numpy as np

from planning_rl.ga.config import GAConfig
from planning_rl.ga.types import GAStats
from planning_rl.ga.utils import normalize_weights


def init_population(*, cfg: GAConfig, num_params: int, rng: np.random.Generator) -> np.ndarray:
    if num_params <= 0:
        raise ValueError("num_params must be >= 1")
    lo, hi = cfg.weight_init_range
    if lo >= hi:
        raise ValueError("weight_init_range must be (lo, hi) with lo < hi")
    pop = rng.uniform(lo, hi, size=(cfg.population_size, num_params)).astype(np.float64)
    if cfg.normalize_weights:
        pop = normalize_weights(pop)
    return pop


def mutate(weights: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
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
        out = normalize_weights(out)
    return out


def crossover(
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


def evolve_population(
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
            child = crossover(
                parent_pool[ia],
                parent_pool[ib],
                float(parent_scores[ia]),
                float(parent_scores[ib]),
                cfg,
                rng,
            )
            child = mutate(child, cfg, rng)
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

            child = crossover(
                population[ia],
                population[ib],
                float(scores_arr[ia]),
                float(scores_arr[ib]),
                cfg,
                rng,
            )
            child = mutate(child, cfg, rng)
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


__all__ = ["crossover", "evolve_population", "init_population", "mutate"]
