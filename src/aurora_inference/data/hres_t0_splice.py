"""Copy selected HRES-T0 times from GCS into a local splice zarr (WB2 layout)."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Sequence
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
from aurora_inference.logging import format_elapsed

__all__ = ["COORD_NAMES", "copy_hres_t0_splice", "write_splice_campaign_attrs"]

COORD_NAMES: tuple[str, ...] = ("latitude", "longitude", "level")
_LOG = logging.getLogger(__name__)
_PROGRESS_EVERY = 10
_KIB = 1024
_MIB = 1024**2
_GIB = 1024**3


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
    n_times = int(time_idx.size)
    n_fields = len(wb2_names)
    expected_decoded = _expected_decoded_bytes(src, wb2_names, n_times)
    _LOG.info(
        "copying %s fields × %s times | expected decoded %s (uncompressed payload)",
        n_fields,
        n_times,
        _format_data_size(expected_decoded),
    )
    copy_t0 = time.perf_counter()
    decoded_bytes = 0
    for field_i, name in enumerate(wb2_names, start=1):
        src_arr = cast(Any, src[name])
        _LOG.info(
            "copying %s (%s/%s) | slab %s",
            name,
            field_i,
            n_fields,
            _format_data_size(_decoded_slab_nbytes(src_arr)),
        )
        decoded_bytes += _copy_array_at_time_indices(
            src,
            dest,
            name,
            time_idx,
            field_i=field_i,
            n_fields=n_fields,
            decoded_before=decoded_bytes,
            expected_decoded=expected_decoded,
            copy_t0=copy_t0,
        )
    write_splice_campaign_attrs(dest, eval_schedule)
    elapsed = time.perf_counter() - copy_t0
    on_disk = _dir_nbytes(out_path)
    _LOG.info(
        "wrote splice %s | Time Elapsed - %s | decoded %s | on disk %s",
        out_path,
        format_elapsed(elapsed),
        _format_data_size(decoded_bytes),
        _format_data_size(on_disk),
    )
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
    *,
    field_i: int,
    n_fields: int,
    decoded_before: int,
    expected_decoded: int,
    copy_t0: float,
) -> int:
    src_arr = cast(Any, src_group[name])
    n_times_out = int(time_idx.size)
    dest_arr = dest_group.create_array(
        name,
        shape=(n_times_out, *src_arr.shape[1:]),
        dtype=src_arr.dtype,
        chunks=(min(int(src_arr.chunks[0]), n_times_out), *src_arr.chunks[1:]),
    )
    copied = 0
    for i, t_idx in enumerate(time_idx):
        slab = np.asarray(src_arr[int(t_idx)])
        dest_arr[i] = slab
        copied += int(slab.nbytes)
        n_done = i + 1
        if n_times_out >= _PROGRESS_EVERY and (
            n_done % _PROGRESS_EVERY == 0 or n_done == n_times_out
        ):
            _LOG.info(
                "copying %s (%s/%s) | times %s/%s | Time Elapsed - %s | decoded %s / %s",
                name,
                field_i,
                n_fields,
                n_done,
                n_times_out,
                format_elapsed(time.perf_counter() - copy_t0),
                _format_data_size(decoded_before + copied),
                _format_data_size(expected_decoded),
            )
    _LOG.info(
        "Completed copy (%s/%s) %s | Time Elapsed - %s | decoded %s / %s",
        field_i,
        n_fields,
        name,
        format_elapsed(time.perf_counter() - copy_t0),
        _format_data_size(decoded_before + copied),
        _format_data_size(expected_decoded),
    )
    return copied


def _decoded_slab_nbytes(array: Any) -> int:
    tail = tuple(int(dim) for dim in array.shape[1:])
    n_elem = 1
    for dim in tail:
        n_elem *= dim
    return n_elem * int(np.dtype(array.dtype).itemsize)


def _expected_decoded_bytes(
    src_group: zarr.Group,
    names: Sequence[str],
    n_times: int,
) -> int:
    total = 0
    for name in names:
        total += n_times * _decoded_slab_nbytes(cast(Any, src_group[name]))
    return total


def _format_data_size(n_bytes: int) -> str:
    value = float(n_bytes)
    if value >= _GIB:
        return f"{value / _GIB:.2f} GiB"
    if value >= _MIB:
        return f"{value / _MIB:.1f} MiB"
    if value >= _KIB:
        return f"{value / _KIB:.1f} KiB"
    return f"{n_bytes} B"


def _dir_nbytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


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
