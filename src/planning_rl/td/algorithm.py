# src/planning_rl/td/algorithm.py
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from planning_rl.algorithms import PlanningAlgorithm
from planning_rl.callbacks import PlanningCallback, wrap_callbacks
from planning_rl.logging import ScalarLogger
from planning_rl.td.ckpt import load_td_checkpoint, save_td_checkpoint
from planning_rl.td.config import TDConfig
from planning_rl.td.learn import learn_td
from planning_rl.td.policy import TDPolicy


class TDAlgorithm(PlanningAlgorithm):
    def __init__(
        self,
        *,
        policy: TDPolicy,
        envs: Sequence[Any] | None = None,
        features: Sequence[str] | None = None,
        cfg: TDConfig,
        device: torch.device,
    ) -> None:
        super().__init__(policy=policy)
        self.envs = list(envs) if envs is not None else None
        self.features = list(features) if features is not None else None
        self.cfg = cfg
        self.device = device
        self.num_timesteps = 0
        self.stats: list[Dict[str, Any]] = []
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def predict(self, *, env: Any | None = None) -> Any:
        if env is None:
            raise ValueError("env is required for TD planning policies")
        return super().predict(env=env)

    def save(self, path: Path) -> Path:
        policy_state = self.policy.state_dict()
        meta = {
            "algo": "td",
            "kind": "td_algo_state",
            "num_timesteps": int(self.num_timesteps),
        }
        cfg = asdict(self.cfg)
        model_state = self.policy.value_model.state_dict()
        optimizer_state = None
        if self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()
        return save_td_checkpoint(
            path=Path(path),
            meta=meta,
            cfg=cfg,
            policy_state=policy_state,
            model_state=model_state,
            optimizer_state=optimizer_state,
            stats=list(self.stats),
        )

    def learn(
        self,
        *,
        total_timesteps: int | None = None,
        callback: PlanningCallback | list[PlanningCallback] | None = None,
        logger: ScalarLogger | None = None,
    ) -> list[Dict[str, Any]]:
        envs_use = list(self.envs) if self.envs is not None else None
        features_use = list(self.features) if self.features is not None else None
        if envs_use is None or features_use is None:
            raise ValueError("TDAlgorithm.learn requires envs and features")

        cfg_use = self.cfg
        if total_timesteps is not None:
            cfg_use = replace(self.cfg, total_timesteps=int(total_timesteps))

        cb = wrap_callbacks(callback)
        if cb is not None:
            cb.init_callback(self)
            cb.on_start(
                num_timesteps=int(self.num_timesteps),
                total_timesteps=int(cfg_use.total_timesteps),
                td_config=cfg_use,
                n_envs=int(cfg_use.n_envs),
                features=list(features_use),
            )
        learn_td(
            algo=self,
            envs=envs_use,
            features=features_use,
            cfg=cfg_use,
            callback=cb,
            logger=logger,
        )
        if cb is not None:
            cb.on_end(
                num_timesteps=int(self.num_timesteps),
                stats=list(self.stats),
            )
        return list(self.stats)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: torch.device,
        policy_builder: Callable[[Dict[str, Any], Dict[str, Any], torch.device], TDPolicy],
        envs: Sequence[Any] | None = None,
        features: Sequence[str] | None = None,
    ) -> "TDAlgorithm":
        data = load_td_checkpoint(Path(path))
        meta = data.get("meta", {})
        if str(meta.get("algo", "")).strip().lower() != "td":
            raise ValueError(f"TDAlgorithm.load: not a TD checkpoint: {path}")

        cfg_raw = data.get("cfg", {})
        cfg = TDConfig(**cfg_raw)

        policy_state = data.get("policy_state", {})
        model_state = data.get("model_state", {})
        policy = policy_builder(policy_state, model_state, device)

        algo = cls(policy=policy, envs=envs, features=features, cfg=cfg, device=device)
        algo.num_timesteps = int(meta.get("num_timesteps", 0))

        stats = data.get("stats", None)
        if isinstance(stats, list):
            algo.stats = list(stats)
        return algo


__all__ = ["TDAlgorithm"]
