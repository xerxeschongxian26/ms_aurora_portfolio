"""Streamed non-finite / field-presence checker for WB2 HRES-T0 zarr stores.

Coverage (timestamp in the time index) stays in ``evaluation_schedule``. This
module only inspects weather arrays: missing field, wrong declared shape, and
non-finite values. It does not impute and does not call Aurora.

Grid identity is an input (``GridSpec``): coords must match or the run aborts.
A single weather array that is missing or wrongly shaped is a catalogue row;
the remaining fields are still scanned.
"""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr

from aurora_inference.data.hres_t0 import (
    _ATMOS_NAME_MAP,
    _SURF_NAME_MAP,
    OUT_SAMPLE_END,
    OUT_SAMPLE_START,
    _read_zarr_array,
    _read_zarr_time_index,
    open_connection_to_gcs,
    open_local_zarr,
)
from aurora_inference.logging import format_elapsed

__all__ = [
    "CATALOGUE_FIELDS",
    "DEFAULT_GRID",
    "FieldSpec",
    "FiniteCheckConfigError",
    "FiniteCheckResult",
    "GridSpec",
    "HoleRow",
    "format_elapsed",
    "format_hole_line",
    "open_hres_t0_store",
    "out_sample_window",
    "resolve_output_paths",
    "run_finite_check",
    "select_scan_times",
]

_LOG = logging.getLogger(__name__)

_COORD_NAMES: frozenset[str] = frozenset({"latitude", "longitude", "level", "time"})
_RETRY_HINTS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "429",
    "503",
    "connection reset",
    "temporarily unavailable",
    "server disconnected",
)

CATALOGUE_FIELDS: tuple[str, ...] = (
    "time",
    "aurora_name",
    "wb2_name",
    "level",
    "kind",
    "n_nonfinite",
    "expected_shape",
    "got_shape",
)

HoleKind = Literal["missing_field", "bad_shape", "nonfinite", "io_error"]
VarGroup = Literal["surf", "atmos"]


class FiniteCheckConfigError(Exception):
    """Window, coords, or store identity is invalid; the scan must not start."""


@dataclass(frozen=True)
class GridSpec:
    """Caller-declared HRES-T0 spatial / level sizes for this store."""

    n_lat: int
    n_lon: int
    n_level: int

    @property
    def surf_shape(self) -> tuple[int, int]:
        return (self.n_lat, self.n_lon)

    @property
    def atmos_shape(self) -> tuple[int, int, int]:
        return (self.n_level, self.n_lat, self.n_lon)


DEFAULT_GRID = GridSpec(n_lat=721, n_lon=1440, n_level=13)


@dataclass(frozen=True)
class FieldSpec:
    aurora_name: str
    wb2_name: str
    group: VarGroup


@dataclass(frozen=True)
class HoleRow:
    kind: HoleKind
    aurora_name: str
    wb2_name: str
    time: pd.Timestamp | None = None
    level: int | None = None
    n_nonfinite: int | None = None
    expected_shape: tuple[int, ...] | None = None
    got_shape: tuple[int, ...] | None = None

    def to_csv_dict(self) -> dict[str, str]:
        time_s = "" if self.time is None else _iso_time(self.time)
        level_s = "" if self.level is None else str(self.level)
        n_s = "" if self.n_nonfinite is None else str(self.n_nonfinite)
        return {
            "time": time_s,
            "aurora_name": self.aurora_name,
            "wb2_name": self.wb2_name,
            "level": level_s,
            "kind": self.kind,
            "n_nonfinite": n_s,
            "expected_shape": format_shape(self.expected_shape),
            "got_shape": format_shape(self.got_shape),
        }


@dataclass(frozen=True)
class FiniteCheckResult:
    catalogue_path: Path
    log_path: Path
    n_times: int
    n_times_with_holes: int
    n_hole_rows: int
    config_seconds: float
    scan_seconds: float
    rows: tuple[HoleRow, ...]


class _StoreSession:
    """Holds a zarr group and can reopen after a flaky GCS read."""

    def __init__(self, store: str) -> None:
        self.store = store
        self.group = open_hres_t0_store(store)

    def reopen(self) -> None:
        self.group = open_hres_t0_store(self.store)


def open_hres_t0_store(store: str | Path) -> zarr.Group:
    """Open a local zarr directory or a ``gs://`` WB2 mapper."""
    text = str(store)
    if text.startswith("gs://"):
        return open_connection_to_gcs(text)
    return open_local_zarr(Path(text))


def out_sample_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive out-sample bounds matching ``OUT_SAMPLE_START`` / ``OUT_SAMPLE_END``."""
    end_inclusive = OUT_SAMPLE_END - pd.Timedelta(hours=6)
    return OUT_SAMPLE_START, end_inclusive


def select_scan_times(
    store_times: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return store timestamps in ``[start, end]``. Both endpoints must exist."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start > end:
        msg = f"start {start} is after end {end}"
        raise FiniteCheckConfigError(msg)
    if start not in store_times:
        msg = f"start {start} is not in the store time index"
        raise FiniteCheckConfigError(msg)
    if end not in store_times:
        msg = f"end {end} is not in the store time index"
        raise FiniteCheckConfigError(msg)
    selected = store_times[(store_times >= start) & (store_times <= end)]
    if selected.empty:
        msg = f"no store times in window {start} .. {end}"
        raise FiniteCheckConfigError(msg)
    return selected


def format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return ""
    return "x".join(str(dim) for dim in shape)


def format_hole_line(row: HoleRow) -> str:
    time_s = "NA" if row.time is None else _iso_time(row.time)
    level_s = "-" if row.level is None else str(row.level)
    n_s = "-" if row.n_nonfinite is None else str(row.n_nonfinite)
    expected = format_shape(row.expected_shape) or "-"
    got = format_shape(row.got_shape) or "-"
    return (
        f"HOLE  {time_s}  {row.aurora_name}  level={level_s}  "
        f"kind={row.kind}  n_nonfinite={n_s}  expected={expected}  got={got}"
    )


def aurora_field_specs() -> tuple[FieldSpec, ...]:
    surf = tuple(
        FieldSpec(aurora_name=key, wb2_name=name, group="surf")
        for key, name in _SURF_NAME_MAP.items()
    )
    atmos = tuple(
        FieldSpec(aurora_name=key, wb2_name=name, group="atmos")
        for key, name in _ATMOS_NAME_MAP.items()
    )
    return surf + atmos


def discover_weather_fields(group: zarr.Group) -> tuple[FieldSpec, ...]:
    """Every non-coordinate array, classified by rank (3=surf, 4=atmos)."""
    reverse_surf = {wb2: aurora for aurora, wb2 in _SURF_NAME_MAP.items()}
    reverse_atmos = {wb2: aurora for aurora, wb2 in _ATMOS_NAME_MAP.items()}
    specs: list[FieldSpec] = []
    for name in _member_names(group):
        if name in _COORD_NAMES:
            continue
        node = cast(Any, group[name])
        ndim = int(node.ndim)
        if ndim == 3:
            specs.append(
                FieldSpec(
                    aurora_name=reverse_surf.get(name, name),
                    wb2_name=name,
                    group="surf",
                )
            )
        elif ndim == 4:
            specs.append(
                FieldSpec(
                    aurora_name=reverse_atmos.get(name, name),
                    wb2_name=name,
                    group="atmos",
                )
            )
    return tuple(specs)


def run_finite_check(
    *,
    store: str,
    output_dir: Path,
    output_name: str,
    grid: GridSpec = DEFAULT_GRID,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    out_sample: bool = False,
    all_vars: bool = False,
    max_retries: int = 4,
    resume: bool = False,
) -> FiniteCheckResult:
    """Validate config, stream one time at a time, write CSV catalogue + log."""
    output_dir = output_dir if output_dir.is_absolute() else Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    catalogue_path, log_path = resolve_output_paths(output_dir, output_name)
    progress_path = catalogue_path.with_suffix(".progress")

    config_t0 = time.perf_counter()
    session = _StoreSession(store)
    store_times = _read_zarr_time_index(session.group)
    start_ts, end_ts, window_label = _resolve_window(start=start, end=end, out_sample=out_sample)
    scan_times = select_scan_times(store_times, start=start_ts, end=end_ts)
    _assert_coords_match_grid(session.group, grid)
    requested = discover_weather_fields(session.group) if all_vars else aurora_field_specs()
    config_holes, scannable = _classify_fields(session.group, requested, grid)
    config_seconds = time.perf_counter() - config_t0

    _log_config(
        store=store,
        window_label=window_label,
        start=start_ts,
        end=end_ts,
        n_scan=len(scan_times),
        n_store=len(store_times),
        all_vars=all_vars,
        grid=grid,
        n_scannable=len(scannable),
        n_requested=len(requested),
        config_ok=len(scannable) > 0,
    )
    if len(scannable) == 0:
        msg = "no scannable weather fields (all missing or wrong shape)"
        raise FiniteCheckConfigError(msg)

    _LOG.info(
        "Configuration Valid | %.1fs | beginning non-finite value checks for %s time steps",
        config_seconds,
        len(scan_times),
    )
    for row in config_holes:
        _LOG.info(
            "will record %s %s at every scan time (store-level)",
            row.aurora_name,
            row.kind,
        )

    done_times = _load_progress(progress_path) if resume else set()
    if not resume and progress_path.exists():
        progress_path.unlink()
    append = resume and catalogue_path.exists()
    catalogue_handle = catalogue_path.open("a" if append else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(catalogue_handle, fieldnames=CATALOGUE_FIELDS)
    all_rows: list[HoleRow] = []
    if not append:
        writer.writeheader()
        catalogue_handle.flush()
    config_rows_written = 0

    times_with_holes: set[pd.Timestamp] = set()
    scan_t0 = time.perf_counter()
    n_times = len(scan_times)
    try:
        for step, stamp in enumerate(scan_times, start=1):
            ts = pd.Timestamp(stamp)
            if ts in done_times:
                _LOG.info(
                    "Completed Check (%s/%s) | Time Elapsed - %s | skipped (resume)",
                    step,
                    n_times,
                    format_elapsed(time.perf_counter() - scan_t0),
                )
                continue
            time_idx = _unique_time_index(store_times, ts)
            step_rows = [replace(row, time=ts) for row in config_holes] + _scan_one_time(
                session,
                time_idx=time_idx,
                stamp=ts,
                fields=scannable,
                grid=grid,
                max_retries=max_retries,
            )
            config_rows_written += len(config_holes)
            for row in step_rows:
                _LOG.info("%s", format_hole_line(row))
                writer.writerow(row.to_csv_dict())
                catalogue_handle.flush()
                all_rows.append(row)
            if step_rows:
                times_with_holes.add(ts)
            _append_progress(progress_path, ts)
            _LOG.info(
                "Completed Check (%s/%s) | Time Elapsed - %s",
                step,
                n_times,
                format_elapsed(time.perf_counter() - scan_t0),
            )
    finally:
        catalogue_handle.close()

    scan_seconds = time.perf_counter() - scan_t0
    _LOG.info("wrote catalogue %s", catalogue_path)
    _LOG.info(
        "%s time steps recorded an invalid value (missing field, shape, or non-finite)",
        len(times_with_holes),
    )
    _LOG.info(
        "catalogue rows=%s (including %s missing/shape rows stamped per time)",
        len(all_rows),
        config_rows_written,
    )
    return FiniteCheckResult(
        catalogue_path=catalogue_path,
        log_path=log_path,
        n_times=n_times,
        n_times_with_holes=len(times_with_holes),
        n_hole_rows=len(all_rows),
        config_seconds=config_seconds,
        scan_seconds=scan_seconds,
        rows=tuple(all_rows),
    )


def _resolve_window(
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    out_sample: bool,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if out_sample:
        start_ts, end_ts = out_sample_window()
        if start is not None or end is not None:
            _LOG.info("out_sample=True; ignoring start/end CLI window")
        return start_ts, end_ts, "out-sample"
    if start is None or end is None:
        msg = "start and end are required unless --out-sample is set"
        raise FiniteCheckConfigError(msg)
    return pd.Timestamp(start), pd.Timestamp(end), "custom"


def _assert_coords_match_grid(group: zarr.Group, grid: GridSpec) -> None:
    lat = _read_zarr_array(group, "latitude")
    lon = _read_zarr_array(group, "longitude")
    level = _read_zarr_array(group, "level")
    got = (int(lat.shape[0]), int(lon.shape[0]), int(level.shape[0]))
    expected = (grid.n_lat, grid.n_lon, grid.n_level)
    if got != expected:
        msg = f"store coords lat/lon/level {got} do not match grid input {expected}"
        raise FiniteCheckConfigError(msg)


def _classify_fields(
    group: zarr.Group,
    requested: Sequence[FieldSpec],
    grid: GridSpec,
) -> tuple[list[HoleRow], tuple[FieldSpec, ...]]:
    names = set(_member_names(group))
    holes: list[HoleRow] = []
    scannable: list[FieldSpec] = []
    for spec in requested:
        if spec.wb2_name not in names:
            holes.append(
                HoleRow(
                    kind="missing_field",
                    aurora_name=spec.aurora_name,
                    wb2_name=spec.wb2_name,
                    expected_shape=_expected_tail(spec, grid),
                )
            )
            continue
        node = cast(Any, group[spec.wb2_name])
        got = tuple(int(dim) for dim in node.shape[1:])
        expected = _expected_tail(spec, grid)
        if got != expected:
            holes.append(
                HoleRow(
                    kind="bad_shape",
                    aurora_name=spec.aurora_name,
                    wb2_name=spec.wb2_name,
                    expected_shape=expected,
                    got_shape=got,
                )
            )
            continue
        scannable.append(spec)
    return holes, tuple(scannable)


def _expected_tail(spec: FieldSpec, grid: GridSpec) -> tuple[int, ...]:
    if spec.group == "surf":
        return grid.surf_shape
    return grid.atmos_shape


def _scan_one_time(
    session: _StoreSession,
    *,
    time_idx: int,
    stamp: pd.Timestamp,
    fields: Sequence[FieldSpec],
    grid: GridSpec,
    max_retries: int,
) -> list[HoleRow]:
    levels = np.asarray(_read_zarr_array(session.group, "level"), dtype=np.int32)
    rows: list[HoleRow] = []
    for spec in fields:
        try:
            array = _read_field_slice(
                session,
                spec=spec,
                time_idx=time_idx,
                max_retries=max_retries,
            )
        except Exception as exc:
            if _is_retryable(exc) or _is_io_error(exc):
                rows.append(
                    HoleRow(
                        kind="io_error",
                        aurora_name=spec.aurora_name,
                        wb2_name=spec.wb2_name,
                        time=stamp,
                        expected_shape=_expected_tail(spec, grid),
                    )
                )
                _LOG.warning("I/O failed for %s at %s: %s", spec.wb2_name, stamp, exc)
                continue
            raise
        expected = _expected_tail(spec, grid)
        got = tuple(int(dim) for dim in array.shape)
        if got != expected:
            rows.append(
                HoleRow(
                    kind="bad_shape",
                    aurora_name=spec.aurora_name,
                    wb2_name=spec.wb2_name,
                    time=stamp,
                    expected_shape=expected,
                    got_shape=got,
                )
            )
            continue
        if spec.group == "surf":
            n_bad = _n_nonfinite(array)
            if n_bad > 0:
                rows.append(
                    HoleRow(
                        kind="nonfinite",
                        aurora_name=spec.aurora_name,
                        wb2_name=spec.wb2_name,
                        time=stamp,
                        n_nonfinite=n_bad,
                        expected_shape=expected,
                        got_shape=got,
                    )
                )
            continue
        for level_idx, level_hpa in enumerate(levels.tolist()):
            slab = array[level_idx]
            n_bad = _n_nonfinite(slab)
            if n_bad <= 0:
                continue
            slab_shape = tuple(int(dim) for dim in slab.shape)
            rows.append(
                HoleRow(
                    kind="nonfinite",
                    aurora_name=spec.aurora_name,
                    wb2_name=spec.wb2_name,
                    time=stamp,
                    level=int(level_hpa),
                    n_nonfinite=n_bad,
                    expected_shape=grid.surf_shape,
                    got_shape=slab_shape,
                )
            )
    return rows


def _read_field_slice(
    session: _StoreSession,
    *,
    spec: FieldSpec,
    time_idx: int,
    max_retries: int,
) -> np.ndarray:
    index: tuple[Any, ...]
    if spec.group == "surf":
        index = (time_idx, slice(None))
    else:
        index = (time_idx, slice(None), slice(None))
    return _read_with_retry(
        session,
        key=spec.wb2_name,
        index=index,
        max_retries=max_retries,
    )


def _read_with_retry(
    session: _StoreSession,
    *,
    key: str,
    index: tuple[Any, ...],
    max_retries: int,
) -> np.ndarray:
    attempts = max(1, max_retries)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _read_zarr_array(session.group, key, index)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == attempts:
                raise
            _LOG.warning(
                "retry %s/%s reading %s: %s",
                attempt,
                attempts,
                key,
                exc,
            )
            backoff = min(2**attempt, 32)
            time.sleep(backoff)
            session.reopen()
    assert last_exc is not None
    raise last_exc


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _RETRY_HINTS)


def _is_io_error(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _n_nonfinite(array: np.ndarray) -> int:
    finite = np.isfinite(array)
    return int(array.size - int(finite.sum()))


def _unique_time_index(times: pd.DatetimeIndex, stamp: pd.Timestamp) -> int:
    loc = times.get_loc(stamp)
    if not isinstance(loc, (int, np.integer)):
        msg = f"{stamp} is not a unique time in the store"
        raise FiniteCheckConfigError(msg)
    return int(loc)


def _member_names(group: zarr.Group) -> tuple[str, ...]:
    return tuple(str(name) for name in group)


def _iso_time(stamp: pd.Timestamp) -> str:
    naive = pd.Timestamp(stamp)
    if naive.tzinfo is not None:
        naive = naive.tz_convert("UTC").tz_localize(None)
    return naive.strftime("%Y-%m-%dT%H:%M:%S")


def resolve_output_paths(output_dir: Path, output_name: str) -> tuple[Path, Path]:
    """Return ``(catalogue.csv, catalogue.log)`` under ``output_dir``."""
    catalogue = _csv_path(output_dir, output_name)
    return catalogue, catalogue.with_suffix(".log")


def _csv_path(output_dir: Path, output_name: str) -> Path:
    name = output_name.strip()
    if not name:
        msg = "output file name must be non-empty"
        raise FiniteCheckConfigError(msg)
    path = Path(name).with_suffix(".csv")
    return output_dir / path.name


def _log_config(
    *,
    store: str,
    window_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_scan: int,
    n_store: int,
    all_vars: bool,
    grid: GridSpec,
    n_scannable: int,
    n_requested: int,
    config_ok: bool,
) -> None:
    var_set = "all-weather" if all_vars else "aurora-9"
    status = "valid" if config_ok else "invalid"
    _LOG.info("store=%s", store)
    _LOG.info(
        "window=%s .. %s (%s) status=%s",
        _iso_time(start),
        _iso_time(end),
        window_label,
        status,
    )
    _LOG.info(
        "times=%s store_times=%s vars=%s scannable=%s/%s",
        n_scan,
        n_store,
        var_set,
        n_scannable,
        n_requested,
    )
    _LOG.info(
        "expected surf=%s atmos=%s",
        format_shape(grid.surf_shape),
        format_shape(grid.atmos_shape),
    )


def _load_progress(path: Path) -> set[pd.Timestamp]:
    if not path.exists():
        return set()
    stamps: set[pd.Timestamp] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            stamps.add(pd.Timestamp(stripped))
    return stamps


def _append_progress(path: Path, stamp: pd.Timestamp) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_iso_time(stamp) + "\n")
