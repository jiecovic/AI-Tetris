# src/tetris_rl/core/training/algorithms/base.py
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Callable


class BaseAlgorithm(ABC):
    def __init__(self, *, policy: Any, env: Any) -> None:
        if policy is None:
            raise ValueError("policy must be provided")
        if env is None:
            raise ValueError("env must be provided")
        self.policy = policy
        self.env = env

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self.policy.predict(*args, **kwargs)

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("learn() must be implemented by subclasses")

    def save(self, path: str | Path) -> None:
        save_fn = getattr(self.policy, "save", None)
        if save_fn is None:
            raise NotImplementedError("policy.save() is not available for this algorithm")
        save_fn(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        policy_loader: Callable[[str | Path], Any],
        **kwargs: Any,
    ) -> "BaseAlgorithm":
        policy = policy_loader(path)
        return cls(policy=policy, **kwargs)


__all__ = ["BaseAlgorithm"]
