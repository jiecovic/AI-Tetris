from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig

from tetris_rl.config.io import to_plain_dict
from tetris_rl.config.datagen_spec import DataGenSpec
from tetris_rl.config.root import DataGenConfig
from tetris_rl.datagen.runner import run_datagen
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.paths import repo_root as find_repo_root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate expert data for RL-Tetris.",
        allow_abbrev=False,
    )
    ap.add_argument("-cfg", "--config-file", dest="config_file", default=None, help="path to a YAML config file")
    ap.add_argument(
        "-c",
        "--config-name",
        dest="config_name",
        default="datagen/codemy1_uniform_noise",
        help="config name (no .yaml)",
    )
    ap.add_argument("-p", "--config-path", dest="config_path", default="configs", help="config directory")
    ap.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides (after --)")
    return ap.parse_args(argv)


def _resolve_config_selection(args: argparse.Namespace) -> tuple[Path, str]:
    if args.config_file:
        p = Path(str(args.config_file)).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--config-file not found: {p}")
        # If the file lives under a "configs" dir, use that as config root so defaults work.
        for parent in p.parents:
            if parent.name == "configs":
                rel = p.relative_to(parent).with_suffix("")
                return parent, rel.as_posix()
        return p.parent, p.stem
    base = Path(str(args.config_path)).expanduser()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    return base, str(args.config_name)


def _normalize_overrides(overrides: Sequence[str]) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg_path, cfg_name = _resolve_config_selection(args)
    overrides = _normalize_overrides(args.overrides)

    if cfg_path.is_absolute():
        with initialize_config_dir(version_base=None, config_dir=str(cfg_path)):
            cfg: DictConfig = compose(config_name=str(cfg_name), overrides=overrides)
    else:
        with initialize(version_base=None, config_path=str(cfg_path)):
            cfg = compose(config_name=str(cfg_name), overrides=overrides)

    cfg_dict = to_plain_dict(cfg)
    data_cfg = DataGenConfig.model_validate(cfg_dict)

    logger = setup_logger(
        name="tetris_rl.datagen",
        use_rich=bool(data_cfg.use_rich),
        level=str(data_cfg.log_level),
    )

    try:
        repo = Path(data_cfg.repo_root).resolve() if str(data_cfg.repo_root).strip() else find_repo_root()
    except Exception:
        logger.exception("[datagen] failed to resolve repo root")
        return 4

    logger.info("[datagen] repo_root=%s", str(repo))

    # Helpful summary
    try:
        ds = data_cfg.dataset
        run = data_cfg.run
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

    spec = DataGenSpec(
        dataset=data_cfg.dataset,
        run=data_cfg.run,
        generation=data_cfg.generation,
        expert=data_cfg.expert,
    )

    try:
        out_dir = run_datagen(spec=spec, cfg=cfg_dict, repo_root=repo, logger=logger)
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
