"""Copy selected HRES-T0 times from GCS into a local splice zarr (WB2 layout)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import zarr

from aurora_inference.config import GCS_STORE_LINK
from aurora_inference.data.hres_t0 import (
    _ATMOS_NAME_MAP,
    _SURF_NAME_MAP,
    _read_zarr_array,
    _read_zarr_time_index,
    open_connection_to_gcs,
)
from aurora_inference.evaluation.evaluation_schedule import EvalSchedule, campaign_attr_payload

__all__ = ["COORD_NAMES", "copy_hres_t0_splice", "write_splice_campaign_attrs"]

COORD_NAMES: tuple[str, ...] = ("latitude", "longitude", "level")


def copy_hres_t0_splice(
    eval_schedule: EvalSchedule,
    *,
    gcs_store_link: str = GCS_STORE_LINK,
) -> Path:
    """Download ``eval_schedule.needed_times`` from WB2 HRES-T0 into ``splice_path``."""
    out_path = _resolve_path(eval_schedule.config.splice_path)
    src = open_connection_to_gcs(gcs_store_link)
    store_times = _read_zarr_time_index(src)
    missing = eval_schedule.needed_times.difference(store_times)
    if not missing.empty:
        preview = ", ".join(str(ts) for ts in missing[:8])
        msg = f"times not in GCS HRES-T0 store: {preview}"
        raise ValueError(msg)

    time_idx = np.asarray(
        [_unique_loc(store_times, ts) for ts in eval_schedule.needed_times],
        dtype=np.int64,
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dest = zarr.open_group(out_path, mode="w")
    if not isinstance(dest, zarr.Group):
        msg = f"expected zarr.Group at {out_path}, got {type(dest)}"
        raise TypeError(msg)

    time_hours = _read_zarr_array(src, "time").astype(np.int64, copy=False)[time_idx]
    dest.create_array("time", data=time_hours, chunks=(time_hours.size,))
    for coord in COORD_NAMES:
        src_coord = cast(Any, src[coord])
        dest.create_array(coord, data=np.asarray(src_coord[:]), chunks=src_coord.chunks)

    wb2_names = tuple(_SURF_NAME_MAP.values()) + tuple(_ATMOS_NAME_MAP.values())
    for name in wb2_names:
        _copy_array_at_time_indices(src, dest, name, time_idx)

    write_splice_campaign_attrs(dest, eval_schedule)
    return out_path


def write_splice_campaign_attrs(group: zarr.Group, eval_schedule: EvalSchedule) -> None:
    """Stamp identification metadata; coverage still uses the ``time`` array."""
    payload = campaign_attr_payload(eval_schedule)
    for key, value in payload.items():
        if isinstance(value, dict):
            group.attrs[key] = json.dumps(value)
        elif isinstance(value, (str, int)):
            group.attrs[key] = value
        else:
            group.attrs[key] = str(value)


def _copy_array_at_time_indices(
    src_group: zarr.Group,
    dest_group: zarr.Group,
    name: str,
    time_idx: np.ndarray,
) -> None:
    src_arr = cast(Any, src_group[name])
    n_times_out = int(time_idx.size)
    dest_arr = dest_group.create_array(
        name,
        shape=(n_times_out, *src_arr.shape[1:]),
        dtype=src_arr.dtype,
        chunks=(min(int(src_arr.chunks[0]), n_times_out), *src_arr.chunks[1:]),
    )
    for i, t_idx in enumerate(time_idx):
        dest_arr[i] = np.asarray(src_arr[int(t_idx)])


def _unique_loc(times: pd.DatetimeIndex, stamp: pd.Timestamp) -> int:
    loc = times.get_loc(stamp)
    if not isinstance(loc, (int, np.integer)):
        msg = f"{stamp} is not a unique time in the HRES-T0 store"
        raise KeyError(msg)
    return int(loc)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path
