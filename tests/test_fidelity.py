"""Variant RMSE helpers (synthetic grids; no Aurora weights)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from aurora_inference.evaluation.fidelity import (
    compute_rmse_score_variant_vs_baseline,
    compute_rmse_score_variant_vs_ground_truth,
)


def _constant_field(value: float, *, name: str = "2t") -> xr.Dataset:
    latitude = np.array([-60.0, 0.0, 60.0], dtype=np.float32)
    longitude = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    return xr.Dataset(
        {name: (("latitude", "longitude"), np.full((3, 4), value, dtype=np.float32))},
        coords={"latitude": latitude, "longitude": longitude},
    )


def test_rmse_variant_vs_baseline_constant_offset() -> None:
    baseline = _constant_field(10.0)
    variant = _constant_field(13.0)
    rmse = compute_rmse_score_variant_vs_baseline(
        baseline=baseline, variant=variant, variables=["2t"]
    )
    np.testing.assert_allclose(float(rmse["2t"]), 3.0)


def test_rmse_variant_vs_ground_truth_constant_offset() -> None:
    truth = _constant_field(11.0)
    variant = _constant_field(13.0)
    rmse = compute_rmse_score_variant_vs_ground_truth(
        truth=truth, variant=variant, variables=["2t"]
    )
    np.testing.assert_allclose(float(rmse["2t"]), 2.0)
