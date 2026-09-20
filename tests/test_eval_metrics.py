"""Latitude-weighted MSE vs the committed z500 crop golden (no WeatherBench2)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_inference.evaluation.metrics import MSE, get_lat_weights

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eval_z500_rmse_on_cropped_grid.npz"


def _z500_datasets(loaded: Mapping[str, np.ndarray]) -> tuple[xr.Dataset, xr.Dataset]:
    latitude = np.asarray(loaded["latitude"])
    longitude = np.asarray(loaded["longitude"])
    coords = {"latitude": latitude, "longitude": longitude}
    forecast = xr.Dataset(
        {"z500": (("latitude", "longitude"), np.asarray(loaded["forecast"]))},
        coords=coords,
    )
    truth = xr.Dataset(
        {"z500": (("latitude", "longitude"), np.asarray(loaded["truth"]))},
        coords=coords,
    )
    return forecast, truth


def test_mse_matches_golden_z500_crop_rmse() -> None:
    blob = np.load(_FIXTURE)
    loaded = {key: np.asarray(blob[key]) for key in blob.files}
    forecast, truth = _z500_datasets(loaded)
    mse = MSE().compute_chunk(forecast, truth)
    rmse = float(np.sqrt(mse["z500"]))
    np.testing.assert_equal(rmse, float(loaded["rmse_wb2"]))


def test_mse_rejects_north_to_south_latitude() -> None:
    blob = np.load(_FIXTURE)
    loaded = {key: np.asarray(blob[key]) for key in blob.files}
    forecast, truth = _z500_datasets(loaded)
    forecast = forecast.sortby("latitude", ascending=False)
    truth = truth.sortby("latitude", ascending=False)
    with pytest.raises(ValueError, match="not increasing"):
        MSE().compute_chunk(forecast, truth)


def test_mse_accumulates_in_fp64_returns_fp32() -> None:
    latitude = np.array([-60.0, 0.0, 60.0], dtype=np.float32)
    longitude = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    coords = {"latitude": latitude, "longitude": longitude}
    forecast_values = np.array(
        [[1.0, 1.0001, 0.9999, 1.0], [1.0, 1.0, 1.0, 1.0], [0.9998, 1.0, 1.0002, 1.0]],
        dtype=np.float32,
    )
    truth_values = np.ones((3, 4), dtype=np.float32)
    forecast = xr.Dataset({"2t": (("latitude", "longitude"), forecast_values)}, coords=coords)
    truth = xr.Dataset({"2t": (("latitude", "longitude"), truth_values)}, coords=coords)

    mse = MSE().compute_chunk(forecast, truth)
    assert mse["2t"].dtype == np.float32

    weights = get_lat_weights(forecast)
    squared = (forecast_values.astype(np.float64) - truth_values.astype(np.float64)) ** 2
    expected = float(
        xr.DataArray(squared, dims=("latitude", "longitude"), coords=coords)
        .weighted(weights)
        .mean(["latitude", "longitude"])
    )
    np.testing.assert_allclose(float(mse["2t"]), np.float32(expected))
