"""RMSE of a variant field against a chosen reference (baseline or truth)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from aurora_inference.evaluation.grids import align_truth_to_forecast
from aurora_inference.evaluation.metrics import MSE

__all__ = [
    "compute_rmse_score_variant_vs_baseline",
    "compute_rmse_score_variant_vs_ground_truth",
]


def compute_rmse_score_variant_vs_baseline(
    *,
    baseline: xr.Dataset,
    variant: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    """Per-variable RMSE of (variant - baseline) at one lead."""
    return _rmse(variant=variant, reference=baseline, variables=variables)


def compute_rmse_score_variant_vs_ground_truth(
    *,
    truth: xr.Dataset,
    variant: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    """Per-variable RMSE of (variant - truth) at one lead."""
    return _rmse(variant=variant, reference=truth, variables=variables)


def _rmse(
    *,
    variant: xr.Dataset,
    reference: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    if not variables:
        msg = "variables must be a non-empty list"
        raise ValueError(msg)
    missing = [
        name
        for name in variables
        if name not in variant.data_vars or name not in reference.data_vars
    ]
    if missing:
        msg = f"variables missing from variant or reference: {missing}"
        raise ValueError(msg)
    aligned = align_truth_to_forecast(variant[variables], reference[variables])
    mse = MSE().compute_chunk(variant[variables], aligned)
    return xr.Dataset({name: np.sqrt(mse[name]) for name in mse.data_vars})
