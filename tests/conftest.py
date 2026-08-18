"""Shared pytest fixtures for aurora-inference tests."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from aurora import Batch, Metadata

from aurora_inference.contract import (
    AURORA_PRETRAINED_SPEC,
    ModelSpec,
    validate_batch,
)
from aurora_inference.data.hres_t0 import HresT0Source, _read_zarr_array
from aurora_inference.data.static_vars import EXPECTED_DIMENSIONS as _STATIC_FULL_SHAPE

_INPUT_TIME_STEPS = 2
_WEATHER_DTYPE = torch.float32
_DEFAULT_INIT_TIME = datetime(2020, 6, 15, 12, 0)  # 1200HRS, 15/06/2020

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
HRES_T0_FIXTURE_ZARR = _FIXTURES_DIR / "hres_t0_slice_small.zarr"
HRES_T0_FIXTURE_STATIC = _FIXTURES_DIR / "hres_t0_static.pickle"
# Second timestep in hres_t0_slice_small.zarr (06:00 + 12:00); metadata.time carries t1.
HRES_T0_FIXTURE_INIT_TIME = datetime(2022, 6, 15, 12, 0)


def _aurora_full_grid_coords() -> tuple[np.ndarray, np.ndarray]:
    """Canonical 0.25° lat/lon axes for the full Aurora static grid (721×1440)."""
    height, width = _STATIC_FULL_SHAPE
    lat = np.linspace(-90.0, 90.0, height, dtype=np.float32)
    lon = np.linspace(0.0, 360.0 - (360.0 / width), width, dtype=np.float32)
    return lat, lon


def _spatial_slice_for_fixture(zarr_data: zarr.Group) -> tuple[slice, slice]:
    """Return ``(lat_slice, lon_slice)`` aligning full static arrays to a fixture zarr group."""
    fixture_lat = _read_zarr_array(zarr_data, "latitude").astype(np.float32)
    fixture_lon = _read_zarr_array(zarr_data, "longitude").astype(np.float32)
    full_lat, full_lon = _aurora_full_grid_coords()

    lat_start = int(np.where(np.isclose(full_lat, fixture_lat[0]))[0][0])
    lon_start = int(np.where(np.isclose(full_lon, fixture_lon[0]))[0][0])
    lat_slice = slice(lat_start, lat_start + len(fixture_lat))
    lon_slice = slice(lon_start, lon_start + len(fixture_lon))

    if not np.allclose(full_lat[lat_slice], fixture_lat):
        msg = (
            "fixture latitude does not align with canonical Aurora grid "
            f"(start index {lat_start}, range [{fixture_lat[0]}, {fixture_lat[-1]}])"
        )
        raise AssertionError(msg)
    if not np.allclose(full_lon[lon_slice], fixture_lon):
        msg = (
            "fixture longitude does not align with canonical Aurora grid "
            f"(start index {lon_start}, range [{fixture_lon[0]}, {fixture_lon[-1]}])"
        )
        raise AssertionError(msg)

    return lat_slice, lon_slice


def open_hres_t0_fixture_zarr(*, path: Path = HRES_T0_FIXTURE_ZARR) -> zarr.Group:
    """Open the committed HRES-T0 zarr slice (offline, no GCS)."""
    result = zarr.open(path, mode="r")
    assert isinstance(result, zarr.Group), f"expected a zarr.Group at {path}, got {type(result)}"
    return result


def load_hres_t0_fixture_static(
    *,
    zarr_data: zarr.Group | None = None,
    path: Path = HRES_T0_FIXTURE_STATIC,
) -> dict[str, np.ndarray]:
    """Load static vars cropped to match ``zarr_data``'s lat/lon extent.

    ``hres_t0_static.pickle`` holds the full 721×1440 HF static fields; the zarr
    fixture is a smaller real slice. Crop by coordinate lookup on the canonical
    Aurora grid so static H/W matches weather H/W after ``HresT0Source.load()``.
    """
    if zarr_data is None:
        zarr_data = open_hres_t0_fixture_zarr()
    lat_slice, lon_slice = _spatial_slice_for_fixture(zarr_data)

    with path.open("rb") as handle:
        full_static = pickle.load(handle)

    return {
        key: np.asarray(full_static[key][lat_slice, lon_slice], dtype=np.float32)
        for key in full_static
    }


def make_hres_t0_source(
    *,
    zarr_data: zarr.Group | None = None,
    static_vars: dict[str, np.ndarray] | None = None,
) -> HresT0Source:
    """Build an ``HresT0Source`` wired to the committed offline fixtures."""
    if zarr_data is None:
        zarr_data = open_hres_t0_fixture_zarr()
    if static_vars is None:
        static_vars = load_hres_t0_fixture_static(zarr_data=zarr_data)
    return HresT0Source(ZARR_DATA=zarr_data, STATIC_VARS=static_vars)


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


@pytest.fixture
def hres_t0_zarr() -> zarr.Group:
    """Committed HRES-T0 zarr slice for offline ``HresT0Source`` tests."""
    return open_hres_t0_fixture_zarr()


@pytest.fixture
def hres_t0_static(hres_t0_zarr: zarr.Group) -> dict[str, np.ndarray]:
    """Static vars cropped to ``hres_t0_zarr``'s lat/lon window."""
    return load_hres_t0_fixture_static(zarr_data=hres_t0_zarr)


@pytest.fixture
def hres_t0_source(hres_t0_zarr: zarr.Group, hres_t0_static: dict[str, np.ndarray]) -> HresT0Source:
    """``HresT0Source`` backed by committed fixtures — no network in CI."""
    return make_hres_t0_source(zarr_data=hres_t0_zarr, static_vars=hres_t0_static)
