"""Shared pytest fixtures for aurora-inference tests."""

from __future__ import annotations

from datetime import datetime

import pytest
import torch
from aurora import Batch, Metadata

from aurora_inference.contract import (
    AURORA_PRETRAINED_SPEC,
    ModelSpec,
    validate_batch,
)

_INPUT_TIME_STEPS = 2
_WEATHER_DTYPE = torch.float32
_DEFAULT_INIT_TIME = datetime(2020, 6, 15, 12, 0)  # 1200HRS, 15/06/2020


def make_valid_batch(
    *,
    height: int = 32,
    width: int = 64,
    batch_size: int = 1,
    spec: ModelSpec = AURORA_PRETRAINED_SPEC,
    init_time: datetime = _DEFAULT_INIT_TIME,
) -> Batch:
    """Build a minimal Batch that satisfies ``validate_batch`` for ``spec``."""
    level_count = len(spec.atmos_levels)
    lat = torch.linspace(90.0, -90.0, height, dtype=torch.float32)
    lon = torch.linspace(0.0, 360.0 - (360.0 / width), width, dtype=torch.float32)

    metadata = Metadata(
        lat=lat,
        lon=lon,
        time=(init_time,) * batch_size,
        atmos_levels=spec.atmos_levels,
    )

    surf_shape = (batch_size, _INPUT_TIME_STEPS, height, width)
    atmos_shape = (batch_size, _INPUT_TIME_STEPS, level_count, height, width)
    static_shape = (height, width)

    surf_vars = {key: torch.zeros(surf_shape, dtype=_WEATHER_DTYPE) for key in spec.surf_vars}
    static_vars = {key: torch.zeros(static_shape, dtype=_WEATHER_DTYPE) for key in spec.static_vars}
    atmos_vars = {key: torch.zeros(atmos_shape, dtype=_WEATHER_DTYPE) for key in spec.atmos_vars}

    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


@pytest.fixture
def valid_batch() -> Batch:
    """A default valid batch for contract tests."""
    batch = make_valid_batch()
    validate_batch(batch, AURORA_PRETRAINED_SPEC)
    return batch
