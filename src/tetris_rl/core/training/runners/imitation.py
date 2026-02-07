# src/tetris_rl/core/training/runners/imitation.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from tetris_rl.core.training.imitation.runner import run_imitation


def run_imitation_phase(
    *,
    cfg: dict[str, Any],
    model: Any,
    imitation_cfg: Any,
    eval_cfg: Any,
    callbacks_cfg: Any,
    run_cfg: Any,
    run_dir: Path,
    repo: Path,
    logger: Any,
    algo_type: str,
) -> None:
    if bool(imitation_cfg.enabled):
        if algo_type in {"ppo", "maskable_ppo"}:
            run_imitation(
                cfg=cfg,
                model=model,
                imitation_cfg=imitation_cfg,
                eval_cfg=eval_cfg,
                callbacks_cfg=callbacks_cfg,
                run_cfg=run_cfg,
                run_dir=run_dir,
                repo=repo,
                logger=logger,
            )
        else:
            logger.info(
                f"[imitation] skipped (enabled=true but algo={algo_type!r} does not support imitation)"
            )
    else:
        logger.info("[imitation] skipped (imitation.enabled=false)")


__all__ = ["run_imitation_phase"]
