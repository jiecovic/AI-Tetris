# src/planning_rl/ga/heuristic.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from tetris_rl.policies.planning_policies.heuristic_policy import HeuristicPlanningPolicy
from tetris_rl.policies.spec import HeuristicSearch, HeuristicSpec, save_heuristic_spec
from planning_rl.ga.algorithm import GAAlgorithm
from planning_rl.ga.config import GAConfig, GAEvalConfig
from planning_rl.ga.types import GAStats
from tetris_rl.envs.factory import make_env_from_cfg


@dataclass(frozen=True)
class HeuristicRunResult:
    spec: HeuristicSpec
    best_score: float
    stats: list[GAStats]


class HeuristicGA:
    def __init__(
        self,
        *,
        policy: HeuristicPlanningPolicy | None = None,
        cfg: Any | None = None,
        features: Sequence[str] | None = None,
        search: HeuristicSearch | None = None,
        env_train: Any | None = None,
        env_eval: Any | None = None,
        ga: GAConfig | None = None,
        eval_cfg: GAEvalConfig | None = None,
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
        eval_cfg = eval_cfg or GAEvalConfig()
        if env_train is None:
            if cfg is None:
                raise ValueError("cfg is required to build env for GA evaluation")
            cfg_env = cfg
            if isinstance(cfg, dict) and "env" not in cfg and "env_train" in cfg:
                cfg_env = dict(cfg)
                cfg_env["env"] = cfg["env_train"]
            env_train = make_env_from_cfg(cfg=cfg_env, seed=int(eval_cfg.seed)).env
        if env_eval is None:
            if isinstance(cfg, dict) and "env_eval" in cfg:
                cfg_eval = dict(cfg)
                cfg_eval["env"] = cfg["env_eval"]
                env_eval = make_env_from_cfg(cfg=cfg_eval, seed=int(eval_cfg.seed)).env
            else:
                env_eval = env_train
        self.env = env_train
        self.eval_env = env_eval
        self.algo = GAAlgorithm(
            policy=self.policy,
            env=self.env,
            eval_env=self.eval_env,
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
    ) -> HeuristicRunResult:
        try:
            stats = self.algo.learn(
                generations=int(generations),
                on_generation=on_generation,
                on_candidate=on_candidate,
            )
            spec = self.policy.build_spec(self.best_weights)
            self._best_spec = spec
            return HeuristicRunResult(
                spec=spec,
                best_score=float(self.algo.best_score),
                stats=stats,
            )
        finally:
            self.env.close()
            if self.eval_env is not self.env:
                self.eval_env.close()

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


HeuristicGATrainer = HeuristicGA


__all__ = [
    "HeuristicGA",
    "HeuristicGATrainer",
    "HeuristicRunResult",
]
