# tests/test_warmup_holes_normalization.py
from __future__ import annotations

import pytest

from tetris_rl.core.game.warmup_params import extract_holes_config


def test_extract_holes_config_accepts_fixed_holes() -> None:
    fixed, rng = extract_holes_config({"holes": 3})
    assert fixed == 3
    assert rng is None


def test_extract_holes_config_accepts_range_holes() -> None:
    fixed, rng = extract_holes_config({"holes": [1, 3]})
    assert fixed == 1
    assert rng == (1, 3)


def test_extract_holes_config_accepts_legacy_uniform_holes() -> None:
    fixed, rng = extract_holes_config({"uniform_holes": {"min": 2, "max": 4}})
    assert fixed == 1
    assert rng == (2, 4)


def test_extract_holes_config_rejects_conflicting_ranges() -> None:
    with pytest.raises(ValueError, match="disagree"):
        extract_holes_config({"holes": {"min": 1, "max": 2}, "uniform_holes": {"min": 2, "max": 3}})
