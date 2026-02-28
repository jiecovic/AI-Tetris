# tests/test_config_schema_guardrails.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tetris_rl.core.config.root import ExperimentConfig, ImitationExperimentConfig


def test_experiment_config_rejects_legacy_policy_init() -> None:
    with pytest.raises(ValidationError, match="policy_init is removed"):
        ExperimentConfig.model_validate(
            {
                "algo": {"type": "maskable_ppo"},
                "policy_init": {"source": "runs/some_run", "which": "latest"},
            }
        )


def test_imitation_config_rejects_legacy_learn_resume() -> None:
    with pytest.raises(ValidationError, match="learn.policy_init/learn.resume are removed"):
        ImitationExperimentConfig.model_validate(
            {
                "algo": {"type": "imitation", "params": {"policy_backend": "maskable_ppo"}},
                "learn": {"resume": None},
            }
        )
