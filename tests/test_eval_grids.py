"""Batch ↔ xarray and HRES-T0 analysis helpers (no Aurora weights)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from conftest import HRES_T0_FIXTURE_INIT_TIME, HRES_T0_FIXTURE_ZARR, make_valid_batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.data.hres_t0 import HresT0Source, open_local_zarr
from aurora_inference.evaluation.grids import (
    align_truth_to_forecast,
    analysis_dataset,
    batch_to_dataset,
    rmse_rows_from_mse,
    score_lead_rows,
)

_T0 = HRES_T0_FIXTURE_INIT_TIME - timedelta(hours=6)
_MISSING = datetime(2022, 6, 15, 18, 0)


def test_batch_to_dataset_uses_last_time_and_increasing_lat() -> None:
    batch = make_valid_batch()
    batch.surf_vars["2t"][0, 0] = 1.0
    batch.surf_vars["2t"][0, 1] = 2.0
    dataset = batch_to_dataset(batch)
    assert dataset["2t"].dims == ("latitude", "longitude")
    assert np.all(np.diff(dataset.latitude.values) > 0)
    np.testing.assert_allclose(dataset["2t"].values, 2.0)


def test_align_truth_to_forecast_crops_one_latitude() -> None:
    forecast = batch_to_dataset(make_valid_batch())
    truth = forecast.copy(deep=True)
    cropped = forecast.isel(latitude=slice(None, -1))
    aligned = align_truth_to_forecast(cropped, truth)
    assert aligned.sizes["latitude"] == cropped.sizes["latitude"]
    np.testing.assert_allclose(aligned.latitude.values, cropped.latitude.values)


def test_analysis_dataset_matches_load_t1(hres_t0_source: HresT0Source) -> None:
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    from_batch = batch_to_dataset(batch)
    from_zarr = analysis_dataset(hres_t0_source, HRES_T0_FIXTURE_INIT_TIME)
    aligned = align_truth_to_forecast(from_batch, from_zarr)
    np.testing.assert_allclose(from_batch["2t"].values, aligned["2t"].values)
    np.testing.assert_allclose(from_batch["z"].values, aligned["z"].values)


def test_analysis_dataset_allows_06_utc(hres_t0_source: HresT0Source) -> None:
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    analysis = analysis_dataset(hres_t0_source, _T0)
    lat_order = np.argsort(np.asarray(batch.metadata.lat.cpu().numpy()))
    expected = np.asarray(batch.surf_vars["2t"][0, 0].cpu().numpy())[lat_order]
    np.testing.assert_allclose(analysis["2t"].values, expected)


def test_analysis_dataset_rejects_missing_time(hres_t0_source: HresT0Source) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        analysis_dataset(hres_t0_source, _MISSING)


def test_score_lead_rows_zero_when_pred_is_truth(hres_t0_source: HresT0Source) -> None:
    batch = hres_t0_source.load(HRES_T0_FIXTURE_INIT_TIME, AURORA_PRETRAINED_SPEC)
    rows = score_lead_rows(
        batch,
        source=hres_t0_source,
        valid_time=HRES_T0_FIXTURE_INIT_TIME,
        init_id=1,
        init_time=HRES_T0_FIXTURE_INIT_TIME,
        lead_hours=0,
    )
    assert rows
    assert all(row["rmse"] == pytest.approx(0.0) for row in rows)


def test_rmse_rows_from_mse_splits_surface_and_levels() -> None:
    mse = xr.Dataset(
        {
            "2t": xr.DataArray(4.0),
            "z": xr.DataArray([9.0, 16.0], dims=("level",), coords={"level": [500, 850]}),
        }
    )
    rows = rmse_rows_from_mse(mse, init_id=1, init_time=HRES_T0_FIXTURE_INIT_TIME, lead_hours=6)
    by_key = {(row["variable"], row["level"]): row["rmse"] for row in rows}
    assert by_key[("2t", None)] == pytest.approx(2.0)
    assert by_key[("z", 500)] == pytest.approx(3.0)
    assert by_key[("z", 850)] == pytest.approx(4.0)


def test_open_local_zarr_reads_fixture() -> None:
    group = open_local_zarr(HRES_T0_FIXTURE_ZARR)
    assert "2m_temperature" in group


def test_open_local_zarr_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="splice not found"):
        open_local_zarr(tmp_path / "missing.zarr")
