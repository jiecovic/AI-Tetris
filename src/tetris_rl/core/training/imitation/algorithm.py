# src/tetris_rl/core/training/imitation/algorithm.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

from tetris_rl.core.training.algorithms.base import BaseAlgorithm
from tetris_rl.core.training.config import ImitationAlgoParams, ImitationLearnConfig
from tetris_rl.core.training.tb_logger import TensorboardLogger

PolicySpec = Union[ActorCriticPolicy, MaskableActorCriticPolicy]
EnvSpec = GymEnv


class ImitationAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        *,
        policy: PolicySpec,
        env: EnvSpec,
        params: ImitationAlgoParams,
    ) -> None:
        if params is None:
            raise ValueError("params must be provided")

        if not isinstance(policy, (ActorCriticPolicy, MaskableActorCriticPolicy)):
            raise TypeError("policy must be an instantiated ActorCriticPolicy or MaskableActorCriticPolicy")

        super().__init__(policy=policy, env=env)
        self.params = params
        self.policy_backend = str(self.params.policy_backend).strip().lower()
        if self.policy_backend == "maskable_ppo" and not isinstance(policy, MaskableActorCriticPolicy):
            raise ValueError("policy_backend=maskable_ppo requires a MaskableActorCriticPolicy instance")
        if self.policy_backend == "ppo" and isinstance(policy, MaskableActorCriticPolicy):
            raise ValueError("policy_backend=ppo requires an ActorCriticPolicy instance")
        self._tetris_algo_type = "maskable_ppo" if isinstance(policy, MaskableActorCriticPolicy) else "ppo"

    def learn(
        self,
        *,
        cfg: Dict[str, Any],
        run_cfg: Any,
        callbacks_cfg: Any,
        learn_cfg: ImitationLearnConfig,
        run_dir: Path,
        repo: Optional[Path] = None,
        logger: Any = None,
        tb_logger: Optional[TensorboardLogger] = None,
    ) -> Dict[str, Any]:
        from tetris_rl.core.training.imitation.trainer import ImitationTrainer

        trainer = ImitationTrainer(
            cfg=cfg,
            algo=self,
            learn_cfg=learn_cfg,
            callbacks_cfg=callbacks_cfg,
            run_cfg=run_cfg,
            run_dir=run_dir,
            repo=repo,
            logger=logger,
            tb_logger=tb_logger,
        )
        return trainer.run()

    def save(self, path: str | Path) -> None:
        self.policy.save(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        env: EnvSpec,
        params: ImitationAlgoParams,
        device: str = "cpu",
        policy_loader: Callable[[str | Path], Any] | None = None,
        **_kwargs: Any,
    ) -> "ImitationAlgorithm":
        if policy_loader is not None:
            policy = policy_loader(path)
        else:
            backend = str(params.policy_backend).strip().lower()
            if backend == "maskable_ppo":
                from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

                policy = MaskableActorCriticPolicy.load(str(path), device=str(device))
            elif backend == "ppo":
                policy = ActorCriticPolicy.load(str(path), device=str(device))
            else:
                raise ValueError(f"unsupported policy_backend: {params.policy_backend!r}")
        return cls(
            policy=policy,
            env=env,
            params=params,
        )


__all__ = ["ImitationAlgorithm"]
