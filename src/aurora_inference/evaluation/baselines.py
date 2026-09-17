"""Persist and reload fp32 baseline forecast fields (WP5b archive).

One lead per zarr directory so a 40-step rollout can write-and-drop. Stage 3
loads these grids for RMSE(variant, baseline). Not a WB2-layout store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import xarray as xr

__all__ = [
    "FIELDS_FULL",
    "FIELDS_HEADLINE",
    "FLOOR_LEAD_HOURS",
    "HEADLINE_SLICES",
    "HeadlineSlice",
    "PersistFields",
    "init_forecast_dir",
    "lead_zarr_path",
    "read_lead_forecast",
    "select_persist_fields",
    "write_lead_forecast",
]

PersistFields = Literal["full", "headline"]

FIELDS_FULL: PersistFields = "full"
FIELDS_HEADLINE: PersistFields = "headline"
FLOOR_LEAD_HOURS: tuple[int, ...] = (24, 120, 240)


@dataclass(frozen=True)
class HeadlineSlice:
    """One Fig. H7 / Q1 headline field: 2-D persist key → Aurora name + optional level."""

    key: str
    aurora_name: str
    level: int | None


HEADLINE_SLICES: tuple[HeadlineSlice, ...] = (
    HeadlineSlice(key="2t", aurora_name="2t", level=None),
    HeadlineSlice(key="10u", aurora_name="10u", level=None),
    HeadlineSlice(key="msl", aurora_name="msl", level=None),
    HeadlineSlice(key="u500", aurora_name="u", level=500),
    HeadlineSlice(key="z500", aurora_name="z", level=500),
    HeadlineSlice(key="t500", aurora_name="t", level=500),
    HeadlineSlice(key="t850", aurora_name="t", level=850),
    HeadlineSlice(key="q500", aurora_name="q", level=500),
)


def init_forecast_dir(output_dir: Path, init_id: int) -> Path:
    """Directory that holds every lead zarr for one init."""
    return output_dir / "baselines" / f"init-{init_id}"


def lead_zarr_path(init_dir: Path, lead_hours: int) -> Path:
    """``baselines/init-N/lead-006.zarr`` (hours, zero-padded to 3)."""
    if lead_hours < 1:
        msg = f"lead_hours must be >= 1 (got {lead_hours})"
        raise ValueError(msg)
    return init_dir / f"lead-{lead_hours:03d}.zarr"


def select_persist_fields(dataset: xr.Dataset, fields: PersistFields) -> xr.Dataset:
    """Subset a one-lead forecast grid. ``headline`` is the eight Q1 2-D slices."""
    if fields == FIELDS_FULL:
        return dataset
    if fields != FIELDS_HEADLINE:
        msg = f"unknown persist fields {fields!r}"
        raise ValueError(msg)
    return _select_headline_fields(dataset)


def headline_aurora_dataset(headline: xr.Dataset, slice_: HeadlineSlice) -> xr.Dataset:
    """One headline 2-D field as an Aurora-named dataset for RMSE vs truth / 5a."""
    if slice_.key not in headline.data_vars:
        msg = f"headline dataset missing {slice_.key!r}"
        raise ValueError(msg)
    array = headline[slice_.key]
    if slice_.level is None:
        return array.to_dataset(name=slice_.aurora_name)
    return array.expand_dims(level=[slice_.level]).to_dataset(name=slice_.aurora_name)


def write_lead_forecast(dataset: xr.Dataset, path: Path) -> None:
    """Write one lead as a zarr directory (overwrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(path, mode="w", consolidated=False)


def read_lead_forecast(path: Path) -> xr.Dataset:
    """Load one persisted lead into memory and close the store."""
    dataset = xr.open_zarr(path, consolidated=False)
    try:
        return cast(xr.Dataset, dataset.load())
    finally:
        dataset.close()


def _select_headline_fields(dataset: xr.Dataset) -> xr.Dataset:
    missing: list[str] = []
    data_vars: dict[str, xr.DataArray] = {}
    for slice_ in HEADLINE_SLICES:
        if slice_.aurora_name not in dataset.data_vars:
            missing.append(slice_.aurora_name)
            continue
        array = dataset[slice_.aurora_name]
        if slice_.level is None:
            data_vars[slice_.key] = array
            continue
        if "level" not in dataset.coords:
            msg = "headline persist needs a level coordinate"
            raise ValueError(msg)
        sliced = array.sel(level=slice_.level)
        if "level" in sliced.coords:
            sliced = sliced.reset_coords("level", drop=True)
        data_vars[slice_.key] = sliced
    if missing:
        msg = f"headline persist needs variables {sorted(set(missing))}"
        raise ValueError(msg)
    return xr.Dataset(data_vars)
