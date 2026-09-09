"""Offline tests for ``HresT0Source`` (WP3 acceptance).

Uses committed fixtures under ``tests/fixtures/`` — no GCS or HuggingFace connection in CI.
See ``.cursor/plans/stage1_forecastpipeline.md`` § WP3 acceptance criteria.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import zarr
from conftest import HRES_T0_FIXTURE_INIT_TIME

from aurora_inference.contract import (
    AURORA_PRETRAINED_SPEC,
    BatchContractError,
    validate_batch,
)
from aurora_inference.data.hres_t0 import HresT0Source, _read_zarr_array

_LEVEL_500_HPA = 500


def test_load_passes_validate_batch(hres_t0_source: HresT0Source) -> None:
    """``load(T, spec)`` returns a batch that passes ``validate_batch`` for the 13-level spec."""
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_orientation_lat_descending_lon_passthrough(
    hres_t0_source: HresT0Source,
    hres_t0_zarr: zarr.Group,
) -> None:
    """Lat is flipped to descending; lon is passthrough; field values match the flipped grid."""
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    raw_lat = _read_zarr_array(hres_t0_zarr, "latitude").astype(np.float32)
    raw_lon = _read_zarr_array(hres_t0_zarr, "longitude").astype(np.float32)

    lat = batch.metadata.lat
    assert torch.all(lat[:-1] > lat[1:]), "metadata.lat must be strictly decreasing"

    assert torch.equal(batch.metadata.lon, torch.tensor(raw_lon, dtype=torch.float32)), (
        "longitude must pass through unchanged from the zarr store"
    )

    # Confirm the flip actually ran: native HRES-T0 lat is ascending, batch lat is not.
    assert raw_lat[0] < raw_lat[-1]
    assert float(lat[0]) == pytest.approx(float(raw_lat[-1]))
    assert float(lat[-1]) == pytest.approx(float(raw_lat[0]))

    # Physical anchor: 2t at t1 on the flipped H axis matches raw zarr with np.flip on H.
    timestep_indices = np.array([0, 1])
    raw_2t_t1 = _read_zarr_array(
        hres_t0_zarr, "2m_temperature", (timestep_indices[1], slice(None), slice(None))
    )
    expected_2t_t1 = np.flip(raw_2t_t1, axis=0).copy()
    loaded_2t_t1 = batch.surf_vars["2t"][0, 1].numpy()
    assert np.allclose(loaded_2t_t1, expected_2t_t1)


def test_static_vars_passthrough_already_descending(
    hres_t0_source: HresT0Source,
    hres_t0_static: dict[str, np.ndarray],
) -> None:
    """ERA5 static pickle is already north-to-south; load must not flip the H axis."""
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    for key in AURORA_PRETRAINED_SPEC.static_vars:
        loaded = batch.static_vars[key].numpy()
        assert np.allclose(loaded, hres_t0_static[key])
        assert not np.allclose(loaded, np.flip(hres_t0_static[key], axis=0))


def test_atmos_levels_and_z500_index(
    hres_t0_source: HresT0Source,
    hres_t0_zarr: zarr.Group,
) -> None:
    """``atmos_levels`` matches spec order; ``z`` at 500 hPa comes from the 500 hPa level."""
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)

    assert batch.metadata.atmos_levels == AURORA_PRETRAINED_SPEC.atmos_levels
    assert batch.metadata.atmos_levels == (
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        850,
        925,
        1000,
    )

    level_idx_500 = AURORA_PRETRAINED_SPEC.atmos_levels.index(_LEVEL_500_HPA)
    store_levels = [int(x) for x in _read_zarr_array(hres_t0_zarr, "level")]
    store_idx_500 = store_levels.index(_LEVEL_500_HPA)
    assert level_idx_500 == store_idx_500

    timestep_indices = np.array([0, 1])
    raw_z500_t1 = _read_zarr_array(
        hres_t0_zarr,
        "geopotential",
        (timestep_indices[1], store_idx_500, slice(None), slice(None)),
    )
    expected_z500_t1 = np.flip(raw_z500_t1, axis=0).copy()
    loaded_z500_t1 = batch.atmos_vars["z"][0, 1, level_idx_500].numpy()
    assert np.allclose(loaded_z500_t1, expected_z500_t1)


def test_mismatch_rejected_missing_static_key(hres_t0_source: HresT0Source) -> None:
    """A batch missing ``slt`` is rejected (match-not-superset)."""
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    static_vars = dict(batch.static_vars)
    del static_vars["slt"]
    batch = replace(batch, static_vars=static_vars)

    with pytest.raises(BatchContractError, match=r"static_vars: missing slt"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_mismatch_rejected_extra_surf_key(hres_t0_source: HresT0Source) -> None:
    """A batch carrying an unexpected extra surf var is rejected (match-not-superset)."""
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    surf_vars = dict(batch.surf_vars)
    surf_vars["extra_var"] = torch.zeros_like(batch.surf_vars["2t"])
    batch = replace(batch, surf_vars=surf_vars)

    with pytest.raises(BatchContractError, match=r"surf_vars: extra extra_var"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)
