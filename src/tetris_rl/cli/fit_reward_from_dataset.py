# src/tetris_rl/cli/fit_reward_from_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np

from tetris_rl.datagen.shard_reader import ShardDataset
from tetris_rl.rewardfit.api import fit_from_dataset
from tetris_rl.rewardfit.models import MODEL_REGISTRY
from tetris_rl.rewardfit.types import (
    ModelKind,
    NormalizeKind,
    ProgressKind,
    RewardFitConfig,
    SortWeightsKind,
    SplitKind,
)
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.paths import repo_root


def _model_choices() -> list[str]:
    return sorted(str(k).lower().strip() for k in MODEL_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fit a delta-reward model from a pre-generated datagen dataset.\n\n"
            "Requires shards recorded with generation.labels.record_rewardfit=true, i.e. shards contain:\n"
            "  legal_mask: (N, A)\n"
            "  phi:        (N, A)\n"
            "  delta:      (N, A, F)\n\n"
            "We define P(a|s) via softmax(phi/tau) over LEGAL actions, then fit:\n"
            "  log P(a|s) - mean_a log P(a|s)  ≈  f(delta(s,a))\n"
            "(centering removes the per-state constant).\n"
        )
    )

    ap.add_argument("--dataset", "-d", type=str, required=True, help="dataset directory (contains manifest.json)")
    ap.add_argument("--shards", type=str, default="", help="comma-separated shard ids to use (default: all)")

    ap.add_argument(
        "--list-features",
        action="store_true",
        help="print available delta feature names from dataset manifest and exit",
    )

    ap.add_argument("--tau", type=float, default=1.0, help="softmax temperature for phi (must be > 0)")
    ap.add_argument("--topk-actions", type=int, default=0, help="top-K legal actions by phi per state (0 = all)")

    ap.add_argument("--max-states", type=int, default=0, help="cap number of states used (0 = all)")
    ap.add_argument("--max-rows", type=int, default=0, help="cap total (state,action) rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="rng seed (subsampling / caps)")

    ap.add_argument("--out", type=str, default="", help="output .npz (default: <dataset_dir>/reward_fit.npz)")

    ap.add_argument("--log-level", type=str, default="info", help="debug|info|warning|error")
    ap.add_argument("--no-rich", action="store_true", help="disable Rich logging")

    ap.add_argument(
        "--progress",
        type=str,
        default="shards",
        choices=["none", "shards", "states"],
        help="progress UI: none | shards | states",
    )

    model_choices = _model_choices()
    ap.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=model_choices,
        help=("model kind from rewardfit registry. " f"available: {', '.join(model_choices)}"),
    )

    ap.add_argument(
        "--no-bias",
        action="store_true",
        help=(
            "fit without intercept term (bias). "
            "Useful if you want a strictly linear form through origin."
        ),
    )

    ap.add_argument(
        "--normalize",
        type=str,
        default="std",
        choices=["none", "maxabs", "l1", "std"],
        help="post-fit normalization for reward magnitude (default: std)",
    )

    # feature selection
    ap.add_argument(
        "--features",
        type=str,
        default="",
        help="comma-separated feature allowlist (use --list-features to see available names)",
    )
    ap.add_argument(
        "--drop-features",
        type=str,
        default="",
        help="comma-separated feature denylist (applied after --features)",
    )

    # split settings
    ap.add_argument(
        "--split",
        type=str,
        default="state",
        choices=["none", "state", "shard"],
        help="how to split train/eval/test (default: state)",
    )
    ap.add_argument("--eval-frac", type=float, default=0.10, help="fraction of states/shards used for eval")
    ap.add_argument("--test-frac", type=float, default=0.0, help="fraction of states/shards used for test")

    ap.add_argument("--print-weights", action="store_true", help="print fitted weights (default: on)")
    ap.add_argument("--no-print-weights", action="store_true", help="disable printing fitted weights")
    ap.add_argument("--print-snippet", action="store_true", help="print copy/paste reward snippet (default: on)")
    ap.add_argument("--no-print-snippet", action="store_true", help="disable printing copy/paste snippet")
    ap.add_argument(
        "--sort-weights",
        type=str,
        default="abs",
        choices=["none", "abs"],
        help="sort printed weights (default: abs)",
    )

    return ap.parse_args()


def _parse_shards_arg(s: str) -> Optional[List[int]]:
    if not str(s).strip():
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else None


def _parse_csv_names(s: str) -> Optional[List[str]]:
    s = str(s or "").strip()
    if not s:
        return None
    out: List[str] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out if out else None


def main() -> int:
    args = parse_args()
    logger = setup_logger(
        name="fit_reward",
        use_rich=(not bool(args.no_rich)),
        level=str(args.log_level),
    )

    model = str(args.model).lower().strip()
    if model not in MODEL_REGISTRY:
        logger.error("unknown model: %r. available: %s", model, ", ".join(_model_choices()))
        return 2

    repo = repo_root()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = (repo / dataset_dir).resolve()

    if not dataset_dir.is_dir():
        logger.error("dataset dir not found: %s", dataset_dir)
        return 2

    # dataset-dependent CLI action: list available features
    if bool(args.list_features):
        ds = ShardDataset(dataset_dir=dataset_dir)
        base_feature_names = list(getattr(ds.manifest, "feature_names", []) or [])
        if not base_feature_names:
            logger.error("manifest.feature_names missing/empty (did you record_rewardfit=true?)")
            return 2

        logger.info("Available delta features (manifest.feature_names):")
        for i, name in enumerate(base_feature_names):
            logger.info("  %2d: %s", int(i), str(name))
        return 0

    out_path = Path(args.out) if args.out.strip() else (dataset_dir / "reward_fit.npz")
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print_weights = (not bool(args.no_print_weights)) or bool(args.print_weights)
    print_snippet = (not bool(args.no_print_snippet)) or bool(args.print_snippet)

    split = str(args.split).lower().strip()
    if split not in {"none", "state", "shard"}:
        logger.error("unknown split: %r (expected: none|state|shard)", split)
        return 2

    cfg = RewardFitConfig(
        tau=float(args.tau),
        topk_actions=int(args.topk_actions),
        max_states=int(args.max_states),
        max_rows=int(args.max_rows),
        seed=int(args.seed),
        shards=_parse_shards_arg(str(args.shards)),
        model=cast(ModelKind, model),
        progress=cast(ProgressKind, args.progress),
        normalize=cast(NormalizeKind, args.normalize),
        print_weights=bool(print_weights),
        print_snippet=bool(print_snippet),
        sort_weights=cast(SortWeightsKind, args.sort_weights),
        features=_parse_csv_names(str(args.features)),
        drop_features=_parse_csv_names(str(args.drop_features)),
        split=cast(SplitKind, split),
        eval_frac=float(args.eval_frac),
        test_frac=float(args.test_frac),
        fit_intercept=(not bool(args.no_bias)),
    )

    try:
        result = fit_from_dataset(
            dataset_dir=dataset_dir,
            cfg=cfg,
            logger=logger,
            use_rich=(not bool(args.no_rich)),
        )
    except (ValueError, FileNotFoundError, RuntimeError):
        logger.exception("[fit_reward] failed")
        return 3

    save_dict: Dict[str, Any] = dict(result.save_dict)
    np.savez_compressed(out_path, **save_dict)
    logger.info("[fit_reward] saved: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
