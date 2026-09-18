"""Persist and reload fp32 baseline forecast fields (WP5b archive).

One lead per zarr directory so a 40-step rollout can write-and-drop. Stage 3
loads these grids for RMSE(variant, baseline). Not a WB2-layout store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from aurora_inference.evaluation.fidelity import compute_rmse_score_variant_vs_baseline
from aurora_inference.evaluation.grids import (
    align_truth_to_forecast,
    as_naive_datetime,
    rmse_rows_from_mse,
)
from aurora_inference.evaluation.metrics import MSE
from aurora_inference.evaluation.tables import load_rmse_rows

__all__ = [
    "FIELDS_FULL",
    "FIELDS_HEADLINE",
    "FLOOR_LEAD_HOURS",
    "HEADLINE_KEYS",
    "HEADLINE_SLICES",
    "HeadlineSlice",
    "PersistFields",
    "headline_aurora_dataset",
    "init_forecast_dir",
    "lead_zarr_path",
    "persist_init_ids_by_time",
    "read_lead_forecast",
    "score_headline_vs_analysis_rows",
    "score_headline_vs_baseline_rows",
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
HEADLINE_KEYS: tuple[str, ...] = tuple(slice_.key for slice_ in HEADLINE_SLICES)


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


def persist_init_ids_by_time(campaign_dir: Path) -> dict[str, int]:
    """Map ``init_time`` ISO strings to persist ``init_id`` from ``rmse_by_init.csv``.

    Screen TOML re-numbers inits 1..6. The n=30 archive keeps the spread
    ``init_id``. Pair on valid init time, not the campaign-local index.
    """
    rows = load_rmse_rows(campaign_dir)
    if not rows:
        msg = f"no rmse_by_init.csv rows under {campaign_dir}"
        raise FileNotFoundError(msg)
    mapping: dict[str, int] = {}
    for row in rows:
        key = as_naive_datetime(pd.Timestamp(row["init_time"])).isoformat()
        init_id = int(row["init_id"])
        previous = mapping.get(key)
        if previous is not None and previous != init_id:
            msg = f"conflicting persist init_id for {key}: {previous} vs {init_id}"
            raise ValueError(msg)
        mapping[key] = init_id
    return mapping


def score_headline_vs_baseline_rows(
    variant: xr.Dataset,
    baseline: xr.Dataset,
    *,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
) -> list[dict[str, Any]]:
    """Per-lead RMSE(variant, baseline) on the eight Q1 headline keys."""
    rmse = compute_rmse_score_variant_vs_baseline(
        baseline=baseline,
        variant=variant,
        variables=list(HEADLINE_KEYS),
    )
    return _rmse_dataset_to_rows(rmse, init_id=init_id, init_time=init_time, lead_hours=lead_hours)


def score_headline_vs_analysis_rows(
    headline: xr.Dataset,
    analysis: xr.Dataset,
    *,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
) -> list[dict[str, Any]]:
    """Per-lead RMSE(variant, HRES-T0) after mapping headline keys back to Aurora names."""
    rows: list[dict[str, Any]] = []
    for slice_ in HEADLINE_SLICES:
        aurora = headline_aurora_dataset(headline, slice_)
        truth_field = analysis[slice_.aurora_name]
        if slice_.level is not None:
            truth_field = truth_field.sel(level=[slice_.level])
        truth = align_truth_to_forecast(aurora, truth_field.to_dataset(name=slice_.aurora_name))
        mse = MSE().compute_chunk(aurora, truth)
        rows.extend(
            rmse_rows_from_mse(
                mse,
                init_id=init_id,
                init_time=init_time,
                lead_hours=lead_hours,
            )
        )
    return rows


def _rmse_dataset_to_rows(
    rmse: xr.Dataset,
    *,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
) -> list[dict[str, Any]]:
    """Flatten an already-sqrt RMSE dataset (fidelity helpers, not ``MSE``)."""
    rows: list[dict[str, Any]] = []
    init_iso = init_time.isoformat()
    for name, data_array in rmse.data_vars.items():
        if "level" in data_array.dims:
            for level in data_array["level"].values:
                rows.append(
                    {
                        "init_id": init_id,
                        "init_time": init_iso,
                        "lead_hours": lead_hours,
                        "variable": name,
                        "level": int(level),
                        "rmse": float(np.asarray(data_array.sel(level=level)).item()),
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
                    "rmse": float(np.asarray(data_array).item()),
                }
            )
    return rows


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
