"""Latitude-weighted MSE / RMSE for Stage 2 skill.

The spatial weights and MSE reduction match WeatherBench2 ``MSE.compute_chunk``
with no region mask (``weatherbench2.metrics``, Apache-2.0, Google LLC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = ["MSE"]


def _assert_increasing(x: np.ndarray) -> None:
    if not (np.diff(x) > 0).all():
        msg = f"array is not increasing: {x}"
        raise ValueError(msg)


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
    pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
    return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
    """Area weight vs latitude: integral of cos(latitude) between cell bounds."""
    bounds = _latitude_cell_bounds(points)
    _assert_increasing(bounds)
    upper = bounds[1:]
    lower = bounds[:-1]
    return np.sin(upper) - np.sin(lower)


def get_lat_weights(ds: xr.Dataset) -> xr.DataArray:
    """Latitude/area weights from the dataset ``latitude`` coordinate."""
    weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
    weights /= np.mean(weights)
    return ds.latitude.copy(data=weights)


def _spatial_average(dataset: xr.Dataset, *, skipna: bool) -> xr.Dataset:
    """Latitude-weighted mean over latitude and longitude."""
    weights = get_lat_weights(dataset)
    return dataset.weighted(weights).mean(["latitude", "longitude"], skipna=skipna)


@dataclass
class MSE:
    """Mean squared error with WeatherBench2 latitude weights (global, no region)."""

    def compute_chunk(
        self,
        forecast: xr.Dataset,
        truth: xr.Dataset,
        skipna: bool = False,
    ) -> xr.Dataset:
        return _spatial_average((forecast - truth) ** 2, skipna=skipna)
