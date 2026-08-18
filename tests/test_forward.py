"""Tests for ``run_rollout``.

Default suite never loads Aurora weights. The CPU rollout with
``AuroraSmallPretrained`` is ``@pytest.mark.slow`` — run with ``make test-slow``.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast
from unittest.mock import MagicMock

import pytest
from aurora import Aurora, Batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.data.synthetic import SyntheticSource
from aurora_inference.inference.forward import run_rollout
from aurora_inference.model.loader import load_model

_GRID_HEIGHT = 32
_GRID_WIDTH = 64
_ROLLOUT_STEPS = 4
_DEVICE = "cpu"
_INIT_TIME = datetime(2022, 1, 1, 12, 0)
_LEVEL_COUNT = len(AURORA_PRETRAINED_SPEC.atmos_levels)


@pytest.mark.parametrize("steps", [0, -1])
def test_run_rollout_rejects_non_positive_steps(steps: int) -> None:
    """``steps < 1`` is rejected before ``validate_batch`` or rollout."""
    model = cast(Aurora, MagicMock())
    batch = cast(Batch, MagicMock())

    with pytest.raises(ValueError, match=r"steps must be >= 1"):
        run_rollout(model, batch, steps)


@pytest.mark.slow
def test_run_rollout_small_model_cpu_returns_one_batch_per_step() -> None:
    """4-step CPU rollout: one batch per step, each with T=1 and the input grid."""
    source = SyntheticSource(height=_GRID_HEIGHT, width=_GRID_WIDTH)
    batch = source.load(_INIT_TIME, AURORA_PRETRAINED_SPEC)
    model = load_model(device=_DEVICE)

    predictions = run_rollout(model, batch, steps=_ROLLOUT_STEPS)

    assert len(predictions) == _ROLLOUT_STEPS
    expected_surf_var_shape = (1, 1, _GRID_HEIGHT, _GRID_WIDTH)
    expected_atmos_var_shape = (1, 1, _LEVEL_COUNT, _GRID_HEIGHT, _GRID_WIDTH)
    for pred in predictions:
        for tensor in pred.surf_vars.values():
            assert tensor.shape == expected_surf_var_shape
        for tensor in pred.atmos_vars.values():
            assert tensor.shape == expected_atmos_var_shape
