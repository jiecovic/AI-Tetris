# src/tetris_rl/cli/gen_expert_data.py
from __future__ import annotations

import argparse
from pathlib import Path

from tetris_rl.config.loader import load_config
from tetris_rl.datagen.runner import run_datagen
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.paths import relpath, repo_root as find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, required=True)
    ap.add_argument("--repo-root", type=str, default="")
    ap.add_argument("--log-level", type=str, default="info")
    ap.add_argument("--no-rich", action="store_true")
    args = ap.parse_args()

    logger = setup_logger(
        name="tetris_rl.datagen",
        use_rich=(not bool(args.no_rich)),
        level=str(args.log_level),
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        logger.error("[datagen] config file not found: %s", str(cfg_path))
        return 2

    try:
        repo = Path(args.repo_root).resolve() if str(args.repo_root).strip() else find_repo_root()
    except Exception:
        logger.exception("[datagen] failed to resolve repo root")
        return 4

    logger.info("[datagen] repo_root=%s", str(repo))
    logger.info("[datagen] config=%s", relpath(cfg_path, base=repo))

    # Load + resolve specs.* (esp specs.env -> {env, game})
    try:
        loaded = load_config(path=cfg_path, domain="datagen")
        spec = loaded.spec
        cfg = loaded.cfg
    except Exception:
        logger.exception("[datagen] failed to load/resolve/parse config: %s", str(cfg_path))
        return 3

    # Helpful summary
    try:
        ds = spec.dataset
        run = spec.run
        out_dir = (Path(repo) / str(ds.out_root) / str(ds.name)).resolve()

        logger.info(
            "[datagen] dataset=%s  out=%s  shards=%d x %d  workers=%d  compression=%s",
            str(ds.name),
            str(out_dir),
            int(ds.shards.num_shards),
            int(ds.shards.shard_steps),
            int(run.num_workers),
            bool(ds.compression),
        )
    except Exception:
        pass

    try:
        out_dir = run_datagen(spec=spec, cfg=cfg, repo_root=repo, logger=logger)
    except KeyboardInterrupt:
        logger.warning("[datagen] interrupted (Ctrl+C)")
        import os
        os._exit(130)  # guaranteed immediate exit (avoids mp/queue teardown hangs on Windows)
    except Exception:
        logger.exception("[datagen] generation failed")
        return 5

    logger.info("[datagen] dataset written to: %s", str(out_dir))
    logger.info("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
