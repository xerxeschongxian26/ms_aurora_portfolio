"""Latitude-weighted MSE vs the committed z500 crop golden (no WeatherBench2)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_inference.evaluation.metrics import MSE

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
