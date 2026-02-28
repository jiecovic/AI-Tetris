# tests/test_policy_bootstrap_errors.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tetris_rl.core.runs.run_resolver import resolve_policy_bootstrap


def test_resolve_policy_bootstrap_surfaces_invalid_run_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "algo:\n"
        "  type: maskable_ppo\n"
        "policy_init:\n"
        "  source: runs/previous\n"
        "  which: latest\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="policy_init is removed"):
        resolve_policy_bootstrap(source=str(run_dir), which="latest")


def test_resolve_policy_bootstrap_surfaces_malformed_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_cfg = repo_root / "examples" / "runs" / "cnn_imitation_run_001" / "config.yaml"
    assert sample_cfg.is_file()

    run_dir = tmp_path / "imitation_run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(sample_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    (ckpt_dir / "latest.zip").write_bytes(b"zip-placeholder")
    (ckpt_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        resolve_policy_bootstrap(source=str(run_dir), which="latest")
