"""
Defines the input contract for Aurora inference in this repository.

Aurora can fail silently on bad input hence rigorous validation is essential.

For full specification, see docs/batch-contract.md.

A Batch is not one tensor. It has three variable groups plus metadata:
metadata is itself, a dataclass

  surf_vars   — dict[str, Tensor], shape (B, T, H, W)
  atmos_vars  — dict[str, Tensor], shape (B, T, L, H, W)
  static_vars — dict[str, Tensor], shape (H, W)  # no batch, no time
  metadata    — lat, lon, time, atmos_levels

The dimension are as follows:
Batch x Time Indices x Pressure Levels x Latitude x Longitude

Each of the variables groups have a different shape because:
- surf_vars are surface fields — no atmospheric level dimension
- atmos_vars span equiangular coords, pressure levels, time, and batch
- static_vars do not vary across time or atmospheric level

Models available:
- Aurora,
- AuroraPretrained
- AuroraSmallPretrained
- Aurora12hPretrained
- AuroraHighRes
- AuroraAirPollution
- AuroraWave

AuroraAirPollution an AuroraWave contain variations in their input contract.
Remainder adhere to the base weather model contract

Common Traps:

  Shapes and ranks
  - Surface tensors: rank 4, shape (B, T, H, W); time dimension T must be exactly 2
  - Atmospheric tensors: rank 5, shape (B, T, L, H, W); T must be exactly 2
  - Static tensors: rank 2, shape (H, W) — no B or T dimensions
  - L (level axis) must equal len(metadata.atmos_levels) for every atmos variable
  - H and W must be consistent across every surf, atmos, and static tensor
  - len(lat) == H and len(lon) == W

  Time semantics
  - Time index 0 = earlier input step (t0); index 1 = current input step (t1)
  - metadata.time has length B; each element is the datetime at index 1 (t1), not t0
  - t0 and t1 must be exactly 6 hours apart — enforced at source construction via
    validate_input_times(t0, t1); not recoverable/obvious from Batch alone
  - Model output Batch has T == 1 (single predicted step), not T == 2

  Coordinates
  - lat strictly decreasing from +90 to -90 (ERA5 often arrives ascending — flip it)
  - lon strictly increasing in [0, 360); must not include 360 (ERA5 often uses [-180, 180])
  - lat/lon dtypes should be float32 or float64 at minimum (float32 required for fields)

  Variables
  - Two different "z" keys: static_vars["z"] is orography; atmos_vars["z"] is
    geopotential at pressure levels — same name, different physical fields
  - Required keys (AuroraPretrained / AuroraSmallPretrained):
      surf:   2t, 10u, 10v, msl
      static: lsm, slt, z
      atmos:  t, u, v, q, z
  - CDS/ERA5 names differ from Aurora short names (e.g. 2m_temperature -> 2t)
  - Missing keys are silently omitted by the model; incorrect keys can KeyError or mis-embed


  Pressure levels
  - metadata.atmos_levels must be exactly, in order:
      50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
  - Order is part of the contract — misordered levels produce silently wrong forecasts
  - L axis of atmos tensors must match this ordering

  Data quality
  - All weather tensors must be float32 (ERA5/xarray often loads float64 — cast explicitly)
  - No NaN or Inf in any surf, atmos, or static tensor

  Validation boundary
  - validate_input_times(t0, t1) at every BatchSource.load() — source seam
  - validate_batch(batch, spec) immediately before model.forward() — inference boundary
  - Do not rely on Aurora's internal asserts as the production gate
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import torch
from aurora import Batch, Metadata

__all__ = [
    "AURORA_PRETRAINED_SPEC",
    "AURORA_PRESSURE_LEVELS",
    "Batch",
    "BatchContractError",
    "Metadata",
    "ModelSpec",
    "validate_batch",
    "validate_input_times",
]

# these levels are dependent on the data source
AURORA_PRESSURE_LEVELS: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

_COORDINATE_DTYPES = (torch.float32, torch.float64)
_INPUT_TIME_STEPS = 2  # len([previous time step, current time step]) = 2
_WEATHER_DTYPE = torch.float32


class BatchContractError(Exception):
    """Raised when a Batch violates the inference input contract."""


@dataclass(frozen=True)
class ModelSpec:
    """Required variables and levels for a specific Aurora checkpoint."""

    surf_vars: tuple[str, ...]
    static_vars: tuple[str, ...]
    atmos_vars: tuple[str, ...]
    atmos_levels: tuple[int, ...]
    input_timestep_hours: int = 6


AURORA_PRETRAINED_SPEC = ModelSpec(
    surf_vars=("2t", "10u", "10v", "msl"),
    static_vars=("lsm", "slt", "z"),
    atmos_vars=("z", "u", "v", "t", "q"),
    atmos_levels=AURORA_PRESSURE_LEVELS,
    input_timestep_hours=6,
)


# Note - define specs for other models here e.g. AuroraAirPollution etc
# Note - what if in a multi-batch input. For a batch of n states, 2 x n matrix of time
def validate_input_times(
    t0: datetime,
    t1: datetime,
    *,
    hours: int = 6,
) -> None:
    """Verify two input timesteps are exactly ``hours`` apart.

    Call from every :class:`~aurora_inference.data.base.BatchSource` at construction
    time.

    Function is separate from validate_batch as
    ``Batch.metadata.time`` carries only *t1*, so spacing cannot be checked
    from a :class:`~aurora.batch.Batch` alone.
    """
    expected = timedelta(hours=hours)  # signed delta i.e. hours = 6, expected = +6 hours
    t_delta = t1 - t0
    if t_delta != expected:
        msg = f"input_times: (received delta {t_delta}, expected {expected}; t0={t0!r}, t1={t1!r})"
        raise BatchContractError(msg)


def validate_batch(batch: Batch, spec: ModelSpec) -> None:
    """Validate ``batch`` against ``spec``. Return ``None`` if valid.

    Raises:
        BatchContractError: With an actionable message naming the offending field.
    """
    _validate_required_keys(batch, spec)
    batch_size = _validate_tensor_shapes(batch, spec)
    _validate_atmos_levels(batch, spec)
    _validate_metadata_time(batch, batch_size)
    _validate_coordinates(batch)
    _validate_weather_tensors(batch, spec)


def _validate_required_keys(batch: Batch, spec: ModelSpec) -> None:
    """Ensure every variable required by ``spec`` is present in ``batch``."""
    groups: tuple[tuple[str, tuple[str, ...], dict[str, torch.Tensor]], ...] = (
        ("surf_vars", spec.surf_vars, batch.surf_vars),
        ("static_vars", spec.static_vars, batch.static_vars),
        ("atmos_vars", spec.atmos_vars, batch.atmos_vars),
    )
    for group_name, required, actual in groups:
        missing = [key for key in required if key not in actual]
        if len(missing) > 0:
            missing_str = ", ".join(missing)
            expected_str = ", ".join(required)
            received_str = "(none)" if len(actual) == 0 else ", ".join(sorted(actual))
            msg = (
                f"{group_name}: missing {missing_str} "
                f"(received keys {{{received_str}}}, expected keys {{{expected_str}}})"
            )
            raise BatchContractError(msg)


def _validate_tensor_shapes(batch: Batch, spec: ModelSpec) -> int:
    """Validate ranks and spatial/time dimensions; return batch size ``B``."""
    reference_key = spec.surf_vars[0]
    reference = batch.surf_vars[reference_key]
    if reference.dim() != 4:
        msg = f"surf_vars['{reference_key}']: rank (received {reference.dim()}, expected 4)"
        raise BatchContractError(msg)

    batch_size, time_steps, height, width = (int(x) for x in reference.shape)
    level_count = len(spec.atmos_levels)
    expected_surf = (batch_size, _INPUT_TIME_STEPS, height, width)
    expected_atmos = (batch_size, _INPUT_TIME_STEPS, level_count, height, width)
    expected_static = (height, width)

    if time_steps != _INPUT_TIME_STEPS:
        msg = (
            f"surf_vars['{reference_key}']: T (received {time_steps}, expected {_INPUT_TIME_STEPS})"
        )
        raise BatchContractError(msg)

    for key in spec.surf_vars:
        tensor = batch.surf_vars[key]
        if tensor.shape != expected_surf:
            msg = f"surf_vars['{key}']: (received {tuple(tensor.shape)}, expected {expected_surf})"
            raise BatchContractError(msg)

    for key in spec.static_vars:
        tensor = batch.static_vars[key]
        if tensor.dim() != 2:
            msg = f"static_vars['{key}']: rank (received {tensor.dim()}, expected 2)"
            raise BatchContractError(msg)
        if tensor.shape != expected_static:
            msg = (
                f"static_vars['{key}']: "
                f"(received {tuple(tensor.shape)}, expected {expected_static})"
            )
            raise BatchContractError(msg)

    for key in spec.atmos_vars:
        tensor = batch.atmos_vars[key]
        if tensor.dim() != 5:
            msg = f"atmos_vars['{key}']: rank (received {tensor.dim()}, expected 5)"
            raise BatchContractError(msg)
        if tensor.shape != expected_atmos:
            msg = (
                f"atmos_vars['{key}']: (received {tuple(tensor.shape)}, expected {expected_atmos})"
            )
            raise BatchContractError(msg)

    return batch_size


def _validate_atmos_levels(batch: Batch, spec: ModelSpec) -> None:
    """Ensure ``metadata.atmos_levels`` matches ``spec`` exactly, including order."""
    received = batch.metadata.atmos_levels
    expected = spec.atmos_levels
    if received != expected:
        msg = f"metadata.atmos_levels: (received {received}, expected {expected})"
        raise BatchContractError(msg)


def _validate_metadata_time(batch: Batch, batch_size: int) -> None:
    """Ensure ``metadata.time`` has one entry per batch element."""
    received = len(batch.metadata.time)
    if received != batch_size:
        msg = f"metadata.time: length (received {received}, expected {batch_size})"
        raise BatchContractError(msg)


def _validate_coordinates(batch: Batch) -> None:
    """Validate ``metadata.lat`` and ``metadata.lon`` orientation and dtypes."""
    lat = batch.metadata.lat
    lon = batch.metadata.lon
    h, w = batch.spatial_shape

    if lat.dtype not in _COORDINATE_DTYPES:
        msg = f"metadata.lat: dtype (received {lat.dtype}, expected {_COORDINATE_DTYPES})"
        raise BatchContractError(msg)
    if lon.dtype not in _COORDINATE_DTYPES:
        msg = f"metadata.lon: dtype (received {lon.dtype}, expected {_COORDINATE_DTYPES})"
        raise BatchContractError(msg)

    if lat.dim() != 1:
        msg = f"metadata.lat: rank (received {lat.dim()}, expected 1)"
        raise BatchContractError(msg)
    if lon.dim() != 1:
        msg = f"metadata.lon: rank (received {lon.dim()}, expected 1)"
        raise BatchContractError(msg)

    if lat.shape[0] != h:
        msg = f"metadata.lat: length (received {lat.shape[0]}, expected {h})"
        raise BatchContractError(msg)
    if lon.shape[0] != w:
        msg = f"metadata.lon: length (received {lon.shape[0]}, expected {w})"
        raise BatchContractError(msg)

    if not torch.all((lat >= -90) & (lat <= 90)):
        lat_min = float(lat.min())
        lat_max = float(lat.max())
        msg = (
            f"metadata.lat: coordinate range (received [{lat_min}, {lat_max}], expected [-90, 90])"
        )
        raise BatchContractError(msg)

    if lat.shape[0] >= 2 and not torch.all(lat[1:] < lat[:-1]):
        msg = "metadata.lat: order (received non-decreasing, expected strictly decreasing)"
        raise BatchContractError(msg)

    if not torch.all((lon >= 0) & (lon < 360)):
        lon_min = float(lon.min())
        lon_max = float(lon.max())
        msg = f"metadata.lon: coordinate range (received [{lon_min}, {lon_max}], expected [0, 360))"
        raise BatchContractError(msg)

    if lon.shape[0] >= 2 and not torch.all(lon[1:] > lon[:-1]):
        msg = "metadata.lon: order (received non-increasing, expected strictly increasing)"
        raise BatchContractError(msg)


def _validate_weather_tensors(batch: Batch, spec: ModelSpec) -> None:
    """Ensure all weather tensors are float32 and finite."""
    groups: tuple[tuple[str, tuple[str, ...], dict[str, torch.Tensor]], ...] = (
        ("surf_vars", spec.surf_vars, batch.surf_vars),
        ("static_vars", spec.static_vars, batch.static_vars),
        ("atmos_vars", spec.atmos_vars, batch.atmos_vars),
    )
    for group_name, required, actual in groups:
        for key in required:
            tensor = actual[key]
            if tensor.dtype != _WEATHER_DTYPE:
                msg = (
                    f"{group_name}['{key}']: dtype "
                    f"(received {tensor.dtype}, expected {_WEATHER_DTYPE})"
                )
                raise BatchContractError(msg)
            if not torch.isfinite(tensor).all():
                msg = f"{group_name}['{key}']: values (received non-finite, expected finite)"
                raise BatchContractError(msg)
