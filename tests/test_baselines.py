"""Baseline-forecast zarr persist helpers (synthetic grids; no Aurora weights)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_inference.evaluation.baselines import (
    FLOOR_LEAD_HOURS,
    HEADLINE_KEYS,
    HEADLINE_SLICES,
    headline_aurora_dataset,
    init_forecast_dir,
    lead_zarr_path,
    persist_init_ids_by_time,
    read_lead_forecast,
    score_headline_vs_analysis_rows,
    score_headline_vs_baseline_rows,
    select_persist_fields,
    write_lead_forecast,
)
from aurora_inference.evaluation.tables import write_rmse_tables


def _forecast_grid() -> xr.Dataset:
    latitude = np.array([-60.0, 0.0, 60.0], dtype=np.float32)
    longitude = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    level = np.array([500, 850, 1000], dtype=np.int32)
    surf = np.full((3, 4), 280.0, dtype=np.float32)
    atmos = np.full((3, 3, 4), 50000.0, dtype=np.float32)
    return xr.Dataset(
        {
            "2t": (("latitude", "longitude"), surf),
            "10u": (("latitude", "longitude"), surf + 1.0),
            "10v": (("latitude", "longitude"), surf + 2.0),
            "msl": (("latitude", "longitude"), surf + 3.0),
            "z": (("level", "latitude", "longitude"), atmos),
            "u": (("level", "latitude", "longitude"), atmos + 1.0),
            "v": (("level", "latitude", "longitude"), atmos + 2.0),
            "t": (("level", "latitude", "longitude"), atmos + 3.0),
            "q": (("level", "latitude", "longitude"), atmos + 4.0),
        },
        coords={"latitude": latitude, "longitude": longitude, "level": level},
    )


def test_lead_zarr_path_pads_hours(tmp_path: Path) -> None:
    init_dir = init_forecast_dir(tmp_path, 7)
    assert lead_zarr_path(init_dir, 6) == init_dir / "lead-006.zarr"
    assert lead_zarr_path(init_dir, 240) == init_dir / "lead-240.zarr"


def test_lead_zarr_path_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="lead_hours"):
        lead_zarr_path(Path("baselines"), 0)


def test_select_persist_fields_full_is_identity() -> None:
    dataset = _forecast_grid()
    selected = select_persist_fields(dataset, "full")
    assert set(selected.data_vars) == set(dataset.data_vars)


def test_select_persist_fields_headline_is_eight_q1_slices() -> None:
    selected = select_persist_fields(_forecast_grid(), "headline")
    assert list(selected.data_vars) == [slice_.key for slice_ in HEADLINE_SLICES]
    assert "10v" not in selected.data_vars
    assert "level" not in selected.dims
    np.testing.assert_allclose(selected["z500"].values, _forecast_grid()["z"].sel(level=500).values)
    np.testing.assert_allclose(selected["t850"].values, _forecast_grid()["t"].sel(level=850).values)


def test_headline_aurora_dataset_restores_level_for_rmse() -> None:
    headline = select_persist_fields(_forecast_grid(), "headline")
    z500_slice = next(slice_ for slice_ in HEADLINE_SLICES if slice_.key == "z500")
    z500 = headline_aurora_dataset(headline, z500_slice)
    assert list(z500.data_vars) == ["z"]
    np.testing.assert_array_equal(z500.level.values, [500])


def test_select_persist_fields_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown persist fields"):
        select_persist_fields(_forecast_grid(), "z500-t850")  # type: ignore[arg-type]


def test_write_read_lead_forecast_round_trips(tmp_path: Path) -> None:
    original = select_persist_fields(_forecast_grid(), "headline")
    path = lead_zarr_path(init_forecast_dir(tmp_path, 0), 24)
    write_lead_forecast(original, path)
    loaded = read_lead_forecast(path)
    xr.testing.assert_allclose(loaded, original)


def test_floor_lead_hours_are_days_1_5_10() -> None:
    assert FLOOR_LEAD_HOURS == (24, 120, 240)


def test_persist_init_ids_by_time_pairs_on_init_time(tmp_path: Path) -> None:
    write_rmse_tables(
        [
            {
                "init_id": 6,
                "init_time": "2022-04-07T00:00:00",
                "lead_hours": 6,
                "variable": "2t",
                "level": None,
                "rmse": 1.0,
            },
            {
                "init_id": 6,
                "init_time": "2022-04-07T00:00:00",
                "lead_hours": 12,
                "variable": "2t",
                "level": None,
                "rmse": 1.1,
            },
        ],
        tmp_path,
    )
    mapping = persist_init_ids_by_time(tmp_path)
    assert mapping == {"2022-04-07T00:00:00": 6}


def test_score_headline_vs_baseline_rows_constant_offset() -> None:
    headline = select_persist_fields(_forecast_grid(), "headline")
    shifted = headline.copy(deep=True)
    shifted["2t"] = headline["2t"] + 4.0
    rows = score_headline_vs_baseline_rows(
        shifted,
        headline,
        init_id=6,
        init_time=datetime(2022, 4, 7, 0, 0),
        lead_hours=6,
    )
    by_var = {row["variable"]: row["rmse"] for row in rows}
    assert set(by_var) == set(HEADLINE_KEYS)
    np.testing.assert_allclose(by_var["2t"], 4.0)
    np.testing.assert_allclose(by_var["z500"], 0.0)


def test_score_headline_vs_analysis_rows_uses_aurora_names() -> None:
    full = _forecast_grid()
    headline = select_persist_fields(full, "headline")
    rows = score_headline_vs_analysis_rows(
        headline,
        full,
        init_id=1,
        init_time=datetime(2022, 1, 1, 12, 0),
        lead_hours=6,
    )
    names = {(row["variable"], row["level"]) for row in rows}
    assert ("2t", None) in names
    assert ("z", 500) in names
    assert ("t", 850) in names
    assert all(float(row["rmse"]) == 0.0 for row in rows)
