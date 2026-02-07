# src/planning_rl/ga/algorithm.py
from __future__ import annotations

from dataclasses import asdict
import io
import json
from pathlib import Path
from typing import Any, Callable, Sequence
import zipfile

import numpy as np

from planning_rl.algorithms import PlanningAlgorithm
from planning_rl.callbacks import PlanningCallback, wrap_callbacks
from planning_rl.ga.config import GAConfig, GAEvalConfig
from planning_rl.ga.operators import evolve_population, init_population
from planning_rl.ga.types import GAStats
from planning_rl.ga.utils import episode_seeds, to_jsonable
from planning_rl.policies import VectorParamPolicy


class GAAlgorithm(PlanningAlgorithm):
    def __init__(
        self,
        *,
        policy: VectorParamPolicy,
        env: Any,
        cfg: GAConfig,
        eval_cfg: GAEvalConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.env = env
        self.cfg = cfg
        self.eval_cfg = eval_cfg or GAEvalConfig()
        self.episode_seeds = episode_seeds(
            base_seed=int(self.eval_cfg.seed),
            episodes=int(self.eval_cfg.episodes),
        )
        self.rng = np.random.default_rng(int(seed if seed is not None else cfg.seed))
        self.population = init_population(cfg=cfg, num_params=policy.num_params, rng=self.rng)
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

        self.population, stats = evolve_population(
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
            seeds = episode_seeds(base_seed=int(cfg.seed), episodes=int(cfg.episodes))
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
        callback: PlanningCallback | None = None,
    ) -> list[float]:
        scores: list[float] = []
        for i in range(self.population.shape[0]):
            weights = self.population[i].tolist()
            score = float(self.evaluate_weights(weights=weights, env=self.env))
            scores.append(score)
            if on_candidate is not None:
                on_candidate(i, score)
            if callback is not None:
                callback.on_event(
                    event="candidate",
                    generation=int(self.generation),
                    candidate_index=int(i),
                    score=float(score),
                    weights=weights,
                )
        return scores

    def learn(
        self,
        *,
        generations: int,
        on_generation: Callable[[GAStats], None] | None = None,
        on_candidate: Callable[[int, float], None] | None = None,
        callback: PlanningCallback | list[PlanningCallback] | None = None,
    ) -> list[GAStats]:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        cb = wrap_callbacks(callback)
        if cb is not None:
            cb.init_callback(self)
            cb.on_start(
                generations=int(generations),
                population_size=int(self.population.shape[0]),
                ga_config=self.cfg,
                eval_config=self.eval_cfg,
            )
        for _ in range(int(generations)):
            if cb is not None:
                cb.on_event(
                    event="generation_start",
                    generation=int(self.generation),
                    population=self.population,
                )
            scores = self.evaluate_population(on_candidate=on_candidate, callback=cb)
            stats = self.tell(scores)
            if on_generation is not None:
                on_generation(stats)
            if cb is not None:
                cb.on_event(
                    event="generation_end",
                    generation=int(stats.generation),
                    stats=stats,
                )
        if cb is not None:
            cb.on_end(
                stats=list(self.stats),
                best_score=float(self.best_score),
                best_weights=self.best_weights.tolist(),
            )
        return list(self.stats)

    def save(self, path: Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "algo": "ga",
            "generation": int(self.generation),
            "best_score": float(self.best_score),
            "cfg": asdict(self.cfg),
            "eval_cfg": asdict(self.eval_cfg),
            "rng_state": to_jsonable(self.rng.bit_generator.state),
        }
        stats = [asdict(s) for s in self.stats]
        policy_state = self.policy.state_dict()

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(to_jsonable(meta), indent=2))
            zf.writestr("stats.json", json.dumps(to_jsonable(stats), indent=2))
            if policy_state:
                zf.writestr("policy_state.json", json.dumps(to_jsonable(policy_state), indent=2))

            buf = io.BytesIO()
            np.save(buf, self.population)
            zf.writestr("population.npy", buf.getvalue())
            buf = io.BytesIO()
            np.save(buf, self.best_weights)
            zf.writestr("best_weights.npy", buf.getvalue())

        return out_path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        policy: VectorParamPolicy,
        env: Any,
    ) -> "GAAlgorithm":
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("meta.json") as fh:
                meta = json.load(fh)
            with zf.open("stats.json") as fh:
                stats_raw = json.load(fh)

            cfg = GAConfig(**meta["cfg"])
            eval_cfg = GAEvalConfig(**meta["eval_cfg"])
            algo = cls(policy=policy, env=env, cfg=cfg, eval_cfg=eval_cfg)

            with zf.open("population.npy") as fh:
                algo.population = np.load(fh)
            with zf.open("best_weights.npy") as fh:
                algo.best_weights = np.load(fh)

            algo.best_score = float(meta.get("best_score", float("-inf")))
            algo.generation = int(meta.get("generation", 0))
            algo.stats = [GAStats(**item) for item in stats_raw]

            rng_state = meta.get("rng_state")
            if rng_state is not None:
                algo.rng = np.random.default_rng()
                algo.rng.bit_generator.state = rng_state

            if "policy_state.json" in zf.namelist():
                with zf.open("policy_state.json") as fh:
                    policy_state = json.load(fh)
                algo.policy.load_state_dict(policy_state)

        return algo


__all__ = ["GAAlgorithm"]
