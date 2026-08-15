"""Tests for ``run_forecast`` (WP4).

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
from aurora_inference.inference.forward import run_forecast
from aurora_inference.model.loader import load_model

_GRID_HEIGHT = 32
_GRID_WIDTH = 64
_FORECAST_STEPS = 4
_INIT_TIME = datetime(2022, 1, 1, 12, 0)
_LEVEL_COUNT = len(AURORA_PRETRAINED_SPEC.atmos_levels)


@pytest.mark.parametrize("steps", [0, -1])
def test_run_forecast_rejects_non_positive_steps(steps: int) -> None:
    """``steps < 1`` is rejected before ``validate_batch`` or rollout."""
    model = cast(Aurora, MagicMock())
    batch = cast(Batch, MagicMock())

    with pytest.raises(ValueError, match=r"steps must be >= 1"):
        run_forecast(model, batch, steps)


@pytest.mark.slow
def test_run_forecast_small_model_cpu_returns_one_batch_per_step() -> None:
    """WP4 acceptance: 4-step CPU rollout, each output has T=1 and the input grid."""
    source = SyntheticSource(height=_GRID_HEIGHT, width=_GRID_WIDTH)
    batch = source.load(_INIT_TIME, AURORA_PRETRAINED_SPEC)
    model = load_model(device="cpu")

    forecasts = run_forecast(model, batch, steps=_FORECAST_STEPS)

    assert len(forecasts) == _FORECAST_STEPS
    expected_surf_var_shape = (1, 1, _GRID_HEIGHT, _GRID_WIDTH)
    expected_atmos_var_shape = (1, 1, _LEVEL_COUNT, _GRID_HEIGHT, _GRID_WIDTH)
    for pred in forecasts:
        for tensor in pred.surf_vars.values():
            assert tensor.shape == expected_surf_var_shape
        for tensor in pred.atmos_vars.values():
            assert tensor.shape == expected_atmos_var_shape
