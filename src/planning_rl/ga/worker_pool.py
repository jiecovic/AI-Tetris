# src/planning_rl/ga/worker_pool.py
from __future__ import annotations

import os
import signal
from dataclasses import asdict
from multiprocessing import TimeoutError as MPTimeoutError
from multiprocessing import current_process, get_context
from typing import Any, Callable, Mapping, Sequence

from planning_rl.ga.config import GAFitnessConfig
from planning_rl.ga.types import GAWorkerFactory
from planning_rl.ga.utils import episode_seeds
from planning_rl.policies import VectorParamPolicy
from planning_rl.utils.seed import seed32_from

_WORKER_FACTORY: GAWorkerFactory | None = None
_WORKER_ENV = None
_WORKER_POLICY: VectorParamPolicy | None = None
_WORKER_SEEDS: list[int] | None = None
_WORKER_FITNESS: GAFitnessConfig | None = None


def _worker_id() -> int:
    try:
        ident = current_process()._identity
    except Exception:
        return 0
    if not ident:
        return 0
    return int(ident[0])


def _init_worker(factory: GAWorkerFactory, fitness_cfg: Mapping[str, Any], seed_base: int) -> None:
    if os.name == "nt":
        try:
            import ctypes

            # Ignore Ctrl+C in worker processes to avoid noisy aborts on Windows.
            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
        except Exception:
            pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_FACTORY, _WORKER_ENV, _WORKER_POLICY, _WORKER_SEEDS, _WORKER_FITNESS
    _WORKER_FACTORY = factory
    _WORKER_FITNESS = GAFitnessConfig(**dict(fitness_cfg))
    # Episode seeds drive the actual RNG because each reset(seed=...) overwrites env RNG.
    # We keep these identical across workers for fair fitness comparisons.
    _WORKER_SEEDS = episode_seeds(
        base_seed=int(seed_base),
        episodes=int(_WORKER_FITNESS.episodes),
    )
    worker_index = max(0, _worker_id() - 1)
    # Worker env seed only matters if reset() is called without a seed.
    env_seed = seed32_from(base_seed=int(seed_base), stream_id=int(0xA5C3 + worker_index))
    _WORKER_ENV = factory.build_env(seed=int(env_seed), worker_index=int(worker_index))
    _WORKER_POLICY = factory.build_policy()


def _evaluate_candidate(weights: Sequence[float]) -> float:
    if _WORKER_FACTORY is None or _WORKER_ENV is None or _WORKER_POLICY is None or _WORKER_SEEDS is None:
        raise RuntimeError("GA worker not initialized")
    if _WORKER_FITNESS is None:
        raise RuntimeError("GA worker fitness config not initialized")
    _WORKER_POLICY.set_params(list(weights))
    total_reward = 0.0
    total_steps = 0
    for seed in _WORKER_SEEDS:
        _obs, _info = _WORKER_ENV.reset(seed=int(seed))
        _ = _obs
        _ = _info
        steps = 0
        while steps < int(_WORKER_FITNESS.max_steps):
            action = _WORKER_POLICY.predict(env=_WORKER_ENV)
            _obs2, reward, terminated, truncated, _info2 = _WORKER_ENV.step(action)
            _ = _obs2
            _ = _info2
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        total_steps += steps
    if _WORKER_FITNESS.fitness_metric == "reward_per_step":
        return float(total_reward) / float(max(1, total_steps))
    return float(total_reward)


def _worker_eval(args: tuple[int, Sequence[float]]) -> tuple[int, float]:
    idx, weights = args
    return int(idx), float(_evaluate_candidate(weights))


class GAWorkerPool:
    def __init__(
        self,
        *,
        factory: GAWorkerFactory,
        workers: int,
        fitness_cfg: GAFitnessConfig,
        seed_base: int | None = None,
    ) -> None:
        self._factory = factory
        self._fitness_cfg = fitness_cfg
        self._seed_base = int(fitness_cfg.seed if seed_base is None else seed_base)
        self._workers = max(1, int(workers))
        ctx = get_context("spawn")
        self._pool = ctx.Pool(
            processes=int(self._workers),
            initializer=_init_worker,
            initargs=(
                self._factory,
                asdict(self._fitness_cfg),
                int(self._seed_base),
            ),
        )

    def _terminate_pool(self) -> None:
        if self._pool is None:
            return
        pool = self._pool
        self._pool = None
        pool.terminate()
        pool.join()

    def close(self) -> None:
        if self._pool is None:
            return
        pool = self._pool
        self._pool = None
        pool.close()
        pool.join()

    def evaluate_population(
        self,
        *,
        weights: Sequence[Sequence[float]],
        on_candidate: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        if self._pool is None:
            raise RuntimeError("GA worker pool is closed")
        tasks = [(int(i), list(w)) for i, w in enumerate(weights)]
        if not tasks:
            return []
        scores = [0.0 for _ in range(len(tasks))]
        pool = self._pool
        prev_handler = signal.getsignal(signal.SIGINT)

        def _sigint_handler(_signum, _frame) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            it = pool.imap_unordered(_worker_eval, tasks)
            remaining = len(tasks)
            # Poll so Ctrl+C is responsive on Windows even when workers run long episodes.
            while remaining > 0:
                try:
                    idx, score = it.next(timeout=0.2)
                except MPTimeoutError:
                    continue
                scores[int(idx)] = float(score)
                if on_candidate is not None:
                    on_candidate(int(idx), float(score))
                remaining -= 1
        except KeyboardInterrupt:
            self._terminate_pool()
            raise
        finally:
            signal.signal(signal.SIGINT, prev_handler)

        return scores


__all__ = ["GAWorkerPool"]
