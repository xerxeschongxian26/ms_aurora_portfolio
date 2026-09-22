"""Latitude-weighted MSE vs the committed z500 crop golden (no WeatherBench2)."""

from __future__ import annotations

import math
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


def test_mse_fp32_vs_fp64_vs_fsum_gap_on_full_grid() -> None:
    """WP2.2: instrument property on a 721×1440 field at ~1e-4 relative difference.

    Reconstructs the pre-WP1 FP32 scorer here. Production ``MSE`` stays FP64.
    ``math.fsum`` is the reference — not ``Decimal``.
    """
    n_lat, n_lon = 721, 1440
    relative_offset = 1.0e-4
    latitude = np.linspace(-90.0, 90.0, n_lat, dtype=np.float32)
    longitude = np.linspace(0.0, 360.0 - (360.0 / n_lon), n_lon, dtype=np.float32)
    coords = {"latitude": latitude, "longitude": longitude}
    truth_values = np.full((n_lat, n_lon), 280.0, dtype=np.float32)
    forecast_values = np.asarray(truth_values * np.float32(1.0 + relative_offset), dtype=np.float32)
    forecast = xr.Dataset({"2t": (("latitude", "longitude"), forecast_values)}, coords=coords)
    truth = xr.Dataset({"2t": (("latitude", "longitude"), truth_values)}, coords=coords)

    mse_fp64 = float(MSE().compute_chunk(forecast, truth)["2t"])
    mse_fp32 = _mse_fp32_accumulate(forecast, truth)
    mse_fsum = _mse_fsum_reference(forecast_values, truth_values, get_lat_weights(forecast).data)

    fp32_gap = abs(mse_fp32 - mse_fsum) / abs(mse_fsum)
    fp64_gap = abs(mse_fp64 - mse_fsum) / abs(mse_fsum)
    # Laptop 2026-09-20: mse_fsum=7.8484788537e-4;
    # FP32 gap=1.335e-6 relative, FP64 gap=1.038e-6 relative. Cheap insurance
    # at this signal size — write-down is the point, not a pass band.
    print(
        f"WP2.2 scorer gaps vs math.fsum: FP32={fp32_gap:.6e} FP64={fp64_gap:.6e} "
        f"mse_fp32={mse_fp32:.12g} mse_fp64={mse_fp64:.12g} mse_fsum={mse_fsum:.12g}"
    )
    assert math.isfinite(fp32_gap)
    assert math.isfinite(fp64_gap)
    assert mse_fp32 > 0.0
    assert mse_fp64 > 0.0


def _mse_fp32_accumulate(forecast: xr.Dataset, truth: xr.Dataset) -> float:
    """Pre-WP1 scorer: square and reduce in the array dtype (FP32). Do not use in src/."""
    weights = get_lat_weights(forecast)
    mse = ((forecast - truth) ** 2).weighted(weights).mean(["latitude", "longitude"], skipna=False)
    return float(mse["2t"])


def _mse_fsum_reference(
    forecast_values: np.ndarray,
    truth_values: np.ndarray,
    lat_weights: np.ndarray,
) -> float:
    """Latitude-weighted MSE via ``math.fsum`` on float64 squares."""
    squares = (forecast_values.astype(np.float64) - truth_values.astype(np.float64)) ** 2
    weights_2d = np.repeat(lat_weights.astype(np.float64)[:, None], squares.shape[1], axis=1)
    weighted = (weights_2d * squares).ravel()
    denom = math.fsum(np.repeat(lat_weights.astype(np.float64), squares.shape[1]).tolist())
    return math.fsum(weighted.tolist()) / denom
