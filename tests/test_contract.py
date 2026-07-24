from dataclasses import replace
from datetime import datetime, timedelta

import pytest
import torch
from conftest import make_valid_batch

from aurora_inference.contract import (
    AURORA_PRETRAINED_SPEC,
    BatchContractError,
    validate_batch,
    validate_input_times,
)


def test_validate_batch_accepts_valid_batch() -> None:
    validate_batch(make_valid_batch(), AURORA_PRETRAINED_SPEC)


def test_validate_input_times_accepts_six_hour_spacing() -> None:
    t0 = datetime(2020, 6, 15, 6, 0)
    t1 = datetime(2020, 6, 15, 12, 0)
    validate_input_times(t0, t1)


def test_validate_input_times_rejects_spacing_shorter_than_six_hours() -> None:
    t0 = datetime(2020, 6, 15, 6, 0)
    t1 = t0 + timedelta(hours=3)

    with pytest.raises(BatchContractError, match=r"input_times: \(received delta"):
        validate_input_times(t0, t1)


def test_validate_input_times_rejects_spacing_longer_than_six_hours() -> None:
    t0 = datetime(2020, 6, 15, 6, 0)
    t1 = t0 + timedelta(hours=12)

    with pytest.raises(BatchContractError, match=r"input_times: \(received delta"):
        validate_input_times(t0, t1)


def test_validate_input_times_rejects_t1_before_t0() -> None:
    t0 = datetime(2020, 6, 15, 6, 0)
    t1 = t0 - timedelta(hours=6)

    with pytest.raises(BatchContractError, match=r"input_times: \(received delta"):
        validate_input_times(t0, t1)


def test_validate_batch_rejects_missing_required_key() -> None:
    batch = make_valid_batch()
    static_vars = dict(batch.static_vars)
    del static_vars["slt"]
    batch = replace(batch, static_vars=static_vars)

    with pytest.raises(BatchContractError, match=r"static_vars: missing slt"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_surf_time_dim_not_two() -> None:
    batch = make_valid_batch()
    surf_vars = {key: tensor[:, :1, :, :] for key, tensor in batch.surf_vars.items()}
    batch = replace(batch, surf_vars=surf_vars)

    with pytest.raises(
        BatchContractError, match=r"surf_vars\['2t'\]: T \(received 1, expected 2\)"
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_atmos_wrong_level_count() -> None:
    batch = make_valid_batch()
    level_count = len(AURORA_PRETRAINED_SPEC.atmos_levels)
    atmos_vars = {
        key: tensor[:, :, : level_count - 1, :, :] for key, tensor in batch.atmos_vars.items()
    }
    batch = replace(batch, atmos_vars=atmos_vars)

    with pytest.raises(BatchContractError, match=r"atmos_vars\['z'\]:"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_static_wrong_rank() -> None:
    batch = make_valid_batch()
    static_vars = dict(batch.static_vars)
    static_vars["lsm"] = static_vars["lsm"].unsqueeze(0)
    batch = replace(batch, static_vars=static_vars)

    with pytest.raises(
        BatchContractError, match=r"static_vars\['lsm'\]: rank \(received 3, expected 2\)"
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_inconsistent_spatial_dims() -> None:
    batch = make_valid_batch()
    height, width = batch.spatial_shape
    atmos_vars = dict(batch.atmos_vars)
    atmos_vars["t"] = torch.zeros(1, 2, len(AURORA_PRETRAINED_SPEC.atmos_levels), height + 1, width)
    batch = replace(batch, atmos_vars=atmos_vars)

    with pytest.raises(BatchContractError, match=r"atmos_vars\['t'\]:"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_lat_length_mismatch() -> None:
    batch = make_valid_batch()
    metadata = replace(batch.metadata, lat=batch.metadata.lat[:-1])
    batch = replace(batch, metadata=metadata)

    with pytest.raises(
        BatchContractError, match=r"metadata.lat: length \(received 31, expected 32\)"
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_ascending_lat() -> None:
    batch = make_valid_batch()
    batch.metadata.lat.copy_(torch.linspace(-90.0, 90.0, 32, dtype=torch.float32))

    with pytest.raises(
        BatchContractError,
        match=r"metadata.lat: order \(received non-decreasing, expected strictly decreasing\)",
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_lon_including_360() -> None:
    batch = make_valid_batch()
    batch.metadata.lon[-1] = 360.0

    with pytest.raises(BatchContractError, match=r"metadata.lon: coordinate range"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_lon_length_mismatch() -> None:
    batch = make_valid_batch()
    metadata = replace(batch.metadata, lon=batch.metadata.lon[:-1])
    batch = replace(batch, metadata=metadata)

    with pytest.raises(
        BatchContractError, match=r"metadata.lon: length \(received 63, expected 64\)"
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_non_increasing_lon() -> None:
    batch = make_valid_batch()
    batch.metadata.lon.copy_(torch.linspace(350.0, 0.0, 64, dtype=torch.float32))

    with pytest.raises(
        BatchContractError,
        match=r"metadata.lon: order \(received non-increasing, expected strictly increasing\)",
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_misordered_atmos_levels() -> None:
    batch = make_valid_batch()
    levels = list(AURORA_PRETRAINED_SPEC.atmos_levels)
    levels[0], levels[1] = levels[1], levels[0]
    metadata = replace(batch.metadata, atmos_levels=tuple(levels))
    batch = replace(batch, metadata=metadata)

    with pytest.raises(BatchContractError, match=r"metadata.atmos_levels:"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_metadata_time_length_mismatch() -> None:
    batch = make_valid_batch(batch_size=2)
    metadata = replace(batch.metadata, time=(datetime(2020, 6, 15, 12, 0),))
    batch = replace(batch, metadata=metadata)

    with pytest.raises(
        BatchContractError, match=r"metadata.time: length \(received 1, expected 2\)"
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_non_float32_weather_tensor() -> None:
    batch = make_valid_batch()
    surf_vars = dict(batch.surf_vars)
    surf_vars["2t"] = surf_vars["2t"].to(torch.float64)
    batch = replace(batch, surf_vars=surf_vars)

    with pytest.raises(BatchContractError, match=r"surf_vars\['2t'\]: dtype"):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)


def test_validate_batch_rejects_nan_in_weather_tensor() -> None:
    batch = make_valid_batch()
    surf_vars = dict(batch.surf_vars)
    tensor = surf_vars["2t"].clone()
    tensor[0, 0, 0, 0] = float("nan")
    surf_vars["2t"] = tensor
    batch = replace(batch, surf_vars=surf_vars)

    with pytest.raises(
        BatchContractError,
        match=r"surf_vars\['2t'\]: values \(received non-finite, expected finite\)",
    ):
        validate_batch(batch, AURORA_PRETRAINED_SPEC)
