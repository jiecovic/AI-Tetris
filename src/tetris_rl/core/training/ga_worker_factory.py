# src/tetris_rl/core/training/ga_worker_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from planning_rl.ga.types import GAWorkerFactory, GymEnv
from tetris_rl.core.envs.factory import make_env_from_cfg
from tetris_rl.core.policies.planning_policies.heuristic_policy import HeuristicPlanningPolicy
from tetris_rl.core.policies.spec import HeuristicSearch


@dataclass(frozen=True)
class TetrisGAWorkerFactory(GAWorkerFactory):
    cfg: dict[str, Any]
    env_cfg: dict[str, Any]
    features: list[str]
    search_cfg: dict[str, Any]

    def _with_env_cfg(self) -> dict[str, Any]:
        out = dict(self.cfg)
        out["env"] = dict(self.env_cfg)
        return out

    def build_env(self, *, seed: int, worker_index: int) -> GymEnv:
        _ = worker_index
        return make_env_from_cfg(cfg=self._with_env_cfg(), seed=int(seed)).env

    def build_policy(self) -> HeuristicPlanningPolicy:
        search = HeuristicSearch.model_validate(dict(self.search_cfg))
        return HeuristicPlanningPolicy(features=list(self.features), search=search)


__all__ = ["TetrisGAWorkerFactory"]
