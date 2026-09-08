"""Offline tests for the HRES-T0 streamed finite checker (no GCS)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import zarr
from conftest import HRES_T0_FIXTURE_INIT_TIME, HRES_T0_FIXTURE_ZARR

from aurora_inference.data.hres_t0 import _read_zarr_array
from aurora_inference.data.hres_t0_finite import (
    FiniteCheckConfigError,
    FiniteCheckResult,
    GridSpec,
    HoleRow,
    format_elapsed,
    format_hole_line,
    run_finite_check,
    select_scan_times,
)

_FIXTURE_START = pd.Timestamp("2022-06-15T06:00:00")
_FIXTURE_END = pd.Timestamp(HRES_T0_FIXTURE_INIT_TIME)
_FIXTURE_GRID = GridSpec(n_lat=32, n_lon=64, n_level=13)
_LEVEL_850 = 850


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "hres_t0.zarr"
    shutil.copytree(HRES_T0_FIXTURE_ZARR, dest)
    return dest


def _run(
    store: Path,
    tmp_path: Path,
    *,
    name: str = "holes.csv",
) -> tuple[FiniteCheckResult, list[dict[str, str]]]:
    result = run_finite_check(
        store=str(store),
        output_dir=tmp_path / "out",
        output_name=name,
        grid=_FIXTURE_GRID,
        start=_FIXTURE_START,
        end=_FIXTURE_END,
        max_retries=1,
    )
    with result.catalogue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return result, rows


def test_select_scan_times_requires_endpoints_in_index() -> None:
    times = pd.DatetimeIndex([_FIXTURE_START, _FIXTURE_END])
    selected = select_scan_times(times, start=_FIXTURE_START, end=_FIXTURE_END)
    assert list(selected) == list(times)
    with pytest.raises(FiniteCheckConfigError, match="not in the store"):
        select_scan_times(times, start=pd.Timestamp("2022-01-01T00"), end=_FIXTURE_END)


def test_format_elapsed_and_hole_line() -> None:
    assert format_elapsed(12) == "12s"
    assert format_elapsed(75) == "1m 15s"
    row = HoleRow(
        kind="nonfinite",
        aurora_name="z",
        wb2_name="geopotential",
        time=pd.Timestamp("2022-01-11T18:00:00"),
        level=850,
        n_nonfinite=3,
        expected_shape=(721, 1440),
        got_shape=(721, 1440),
    )
    line = format_hole_line(row)
    assert "HOLE  2022-01-11T18:00:00  z  level=850" in line
    assert "kind=nonfinite" in line
    assert "n_nonfinite=3" in line


def test_clean_fixture_writes_header_only(tmp_path: Path) -> None:
    store = _copy_fixture(tmp_path)
    result, rows = _run(store, tmp_path)
    assert rows == []
    assert result.n_times == 2
    assert result.n_times_with_holes == 0
    assert result.n_hole_rows == 0


def test_wrong_grid_aborts_before_scan(tmp_path: Path) -> None:
    store = _copy_fixture(tmp_path)
    with pytest.raises(FiniteCheckConfigError, match="do not match grid"):
        run_finite_check(
            store=str(store),
            output_dir=tmp_path / "out",
            output_name="holes.csv",
            grid=GridSpec(n_lat=721, n_lon=1440, n_level=13),
            start=_FIXTURE_START,
            end=_FIXTURE_END,
            max_retries=1,
        )


def test_out_sample_requires_year_endpoints(tmp_path: Path) -> None:
    store = _copy_fixture(tmp_path)
    with pytest.raises(FiniteCheckConfigError, match="not in the store"):
        run_finite_check(
            store=str(store),
            output_dir=tmp_path / "out",
            output_name="holes.csv",
            grid=_FIXTURE_GRID,
            out_sample=True,
            max_retries=1,
        )


def test_nan_at_z850_is_catalogued(tmp_path: Path) -> None:
    store = _copy_fixture(tmp_path)
    group = zarr.open_group(store, mode="r+")
    assert isinstance(group, zarr.Group)
    levels = _read_zarr_array(group, "level").astype(np.int32)
    level_idx = int(np.where(levels == _LEVEL_850)[0][0])
    z_arr = cast(Any, group["geopotential"])
    slab = np.asarray(z_arr[1, level_idx])
    slab[0, 0] = np.nan
    z_arr[1, level_idx] = slab

    result, rows = _run(store, tmp_path)
    assert result.n_times_with_holes == 1
    match = [row for row in rows if row["kind"] == "nonfinite"]
    assert len(match) == 1
    row = match[0]
    assert row["time"] == "2022-06-15T12:00:00"
    assert row["aurora_name"] == "z"
    assert row["wb2_name"] == "geopotential"
    assert row["level"] == "850"
    assert row["n_nonfinite"] == "1"


def test_inf_on_surface_is_nonfinite(tmp_path: Path) -> None:
    store = _copy_fixture(tmp_path)
    group = zarr.open_group(store, mode="r+")
    assert isinstance(group, zarr.Group)
    t2_arr = cast(Any, group["2m_temperature"])
    slab = np.asarray(t2_arr[0])
    slab[2, 3] = np.inf
    t2_arr[0] = slab

    _, rows = _run(store, tmp_path)
    match = [row for row in rows if row["aurora_name"] == "2t"]
    assert len(match) == 1
    assert match[0]["kind"] == "nonfinite"
    assert match[0]["time"] == "2022-06-15T06:00:00"
    assert match[0]["n_nonfinite"] == "1"
    assert match[0]["level"] == ""


def test_missing_field_and_bad_shape_still_scan_others(tmp_path: Path) -> None:
    store = _copy_fixture(tmp_path)
    group = zarr.open_group(store, mode="r+")
    assert isinstance(group, zarr.Group)
    del group["10m_u_component_of_wind"]
    del group["mean_sea_level_pressure"]
    n_times = int(_read_zarr_array(group, "time").shape[0])
    group.create_array(
        "mean_sea_level_pressure",
        data=np.zeros((n_times, 8, 8), dtype=np.float32),
        chunks=(1, 8, 8),
    )
    levels = _read_zarr_array(group, "level").astype(np.int32)
    level_idx = int(np.where(levels == _LEVEL_850)[0][0])
    z_arr = cast(Any, group["geopotential"])
    slab = np.asarray(z_arr[1, level_idx])
    slab[1, 1] = np.nan
    z_arr[1, level_idx] = slab

    result, rows = _run(store, tmp_path)
    missing = [row for row in rows if row["kind"] == "missing_field"]
    shaped = [row for row in rows if row["kind"] == "bad_shape"]
    nan_rows = [row for row in rows if row["kind"] == "nonfinite"]
    assert {row["aurora_name"] for row in missing} == {"10u"}
    assert {row["aurora_name"] for row in shaped} == {"msl"}
    assert all(row["time"] for row in missing + shaped)
    assert {row["time"] for row in missing} == {
        "2022-06-15T06:00:00",
        "2022-06-15T12:00:00",
    }
    assert len(missing) == 2
    assert len(shaped) == 2
    assert len(nan_rows) == 1
    assert nan_rows[0]["aurora_name"] == "z"
    assert result.n_times_with_holes == 2
    assert result.n_hole_rows == 5
    assert all(row["time"] for row in rows)
