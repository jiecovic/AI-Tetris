# tests/test_legacy_run_config_loading.py
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tetris_rl.core.config.io import load_imitation_config
from tetris_rl.core.runs.run_resolver import load_run_spec


def test_strict_imitation_loader_rejects_legacy_resume_field() -> None:
    cfg_path = Path("examples/runs/cnn_imitation_run_001/config.yaml")

    with pytest.raises(ValidationError, match="learn.policy_init/learn.resume are removed"):
        load_imitation_config(cfg_path)


def test_run_spec_loader_accepts_legacy_imitation_run_config() -> None:
    spec = load_run_spec("examples/runs/cnn_imitation_run_001")

    assert spec.algo_type == "imitation"
    assert spec.exp_cfg is not None
    assert "resume" not in spec.cfg_plain["learn"]
