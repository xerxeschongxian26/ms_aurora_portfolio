"""Convert Aurora ``Batch`` / HRES-T0 analysis slices to lat-lon ``xarray`` datasets."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from aurora import Batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec
from aurora_inference.data.hres_t0 import (
    _ATMOS_NAME_MAP,
    _SURF_NAME_MAP,
    HresT0Source,
    _read_zarr_array,
)
from aurora_inference.evaluation.metrics import MSE

__all__ = [
    "align_truth_to_forecast",
    "analysis_dataset",
    "as_naive_datetime",
    "batch_to_dataset",
    "rmse_rows_from_mse",
    "score_lead_rows",
]


def as_naive_datetime(value: datetime | pd.Timestamp) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return datetime(
        int(stamp.year),
        int(stamp.month),
        int(stamp.day),
        int(stamp.hour),
        int(stamp.minute),
        int(stamp.second),
    )


def _scalar(value: object) -> float:
    return float(np.asarray(value).item())


def batch_to_dataset(batch: Batch) -> xr.Dataset:
    """T=1 (or last time index) weather fields as an ``xr.Dataset``, lat increasing."""
    lat = np.asarray(batch.metadata.lat.detach().cpu().numpy(), dtype=np.float32)
    lon = np.asarray(batch.metadata.lon.detach().cpu().numpy(), dtype=np.float32)
    levels = [int(level) for level in batch.metadata.atmos_levels]
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for key, tensor in batch.surf_vars.items():
        data_vars[key] = (
            ("latitude", "longitude"),
            np.asarray(tensor[0, -1].detach().cpu().numpy(), dtype=np.float32),
        )
    for key, tensor in batch.atmos_vars.items():
        data_vars[key] = (
            ("level", "latitude", "longitude"),
            np.asarray(tensor[0, -1].detach().cpu().numpy(), dtype=np.float32),
        )
    return xr.Dataset(
        data_vars,
        coords={
            "latitude": lat,
            "longitude": lon,
            "level": np.asarray(levels, dtype=np.int32),
        },
    ).sortby("latitude")


def analysis_dataset(
    source: HresT0Source,
    valid_time: datetime,
    spec: ModelSpec = AURORA_PRETRAINED_SPEC,
) -> xr.Dataset:
    """One HRES-T0 analysis time from the source zarr (native ascending latitude)."""
    stamp = as_naive_datetime(valid_time)
    indices = source._get_timestamp_indices([stamp], source.ZARR_DATA)
    if indices.size != 1:
        msg = f"expected exactly one zarr time for {stamp.isoformat()}, got {indices.size}"
        raise ValueError(msg)
    time_index = int(indices[0])
    lat = _read_zarr_array(source.ZARR_DATA, "latitude").astype(np.float32, copy=False)
    lon = _read_zarr_array(source.ZARR_DATA, "longitude").astype(np.float32, copy=False)
    levels = np.asarray(_read_zarr_array(source.ZARR_DATA, "level"), dtype=np.int32)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for key in spec.surf_vars:
        array = _read_zarr_array(source.ZARR_DATA, _SURF_NAME_MAP[key], (time_index, slice(None)))
        data_vars[key] = (("latitude", "longitude"), np.asarray(array, dtype=np.float32))
    for key in spec.atmos_vars:
        array = _read_zarr_array(
            source.ZARR_DATA,
            _ATMOS_NAME_MAP[key],
            (time_index, slice(None), slice(None)),
        )
        data_vars[key] = (("level", "latitude", "longitude"), np.asarray(array, dtype=np.float32))
    return xr.Dataset(
        data_vars,
        coords={"latitude": lat, "longitude": lon, "level": levels},
    ).sortby("latitude")


def align_truth_to_forecast(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """Select truth onto the forecast grid (Aurora may crop one latitude)."""
    return truth.sel(latitude=forecast.latitude, longitude=forecast.longitude)


def rmse_rows_from_mse(
    mse: xr.Dataset,
    *,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
) -> list[dict[str, object]]:
    """Flatten spatial MSE to long-form RMSE rows (one per surf var or atmos level)."""
    rows: list[dict[str, object]] = []
    init_iso = init_time.isoformat()
    for name, data_array in mse.data_vars.items():
        rmse = cast(xr.DataArray, np.sqrt(data_array))
        if "level" in rmse.dims:
            for level in rmse["level"].values:
                rows.append(
                    {
                        "init_id": init_id,
                        "init_time": init_iso,
                        "lead_hours": lead_hours,
                        "variable": name,
                        "level": int(level),
                        "rmse": _scalar(rmse.sel(level=level)),
                    }
                )
        else:
            rows.append(
                {
                    "init_id": init_id,
                    "init_time": init_iso,
                    "lead_hours": lead_hours,
                    "variable": name,
                    "level": None,
                    "rmse": _scalar(rmse),
                }
            )
    return rows


def score_lead_rows(
    pred: Batch,
    *,
    source: HresT0Source,
    valid_time: datetime,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
    spec: ModelSpec = AURORA_PRETRAINED_SPEC,
) -> list[dict[str, object]]:
    """RMSE of one T=1 pred vs the analysis at ``valid_time`` (lat-weighted, then sqrt)."""
    forecast = batch_to_dataset(pred)
    truth = align_truth_to_forecast(forecast, analysis_dataset(source, valid_time, spec))
    shared = [name for name in forecast.data_vars if name in truth.data_vars]
    mse = MSE().compute_chunk(forecast[shared], truth[shared])
    return rmse_rows_from_mse(mse, init_id=init_id, init_time=init_time, lead_hours=lead_hours)
