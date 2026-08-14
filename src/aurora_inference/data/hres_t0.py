"""BatchSource for the HRES_T0 dataset from WeatherBench2 Google Cloud Storage. The dataset
is stored as a zarray with the format .zarr

It produces Batch objects that satisfy the input contract for the Aurora model.

This dataset spans the period 2016 - 2022 and is the underlying data source
used to fine-tune the pre-trained model, AuroraPreTrained, to produce the fine-tuned model, Aurora

- Time resolution every 6 hours
    - Time stamps have hour values in [00, 06, 12, 18]
    - Earliest available date: 2016-01-01T00:00:00.000000000
    - Latest available date: 2023-01-10T18:00:00.000000000
- In-Sample Time Stamps (used to fine-tune Aurora 0.25 Pretrained)
    - 2016-01-01T00:00:00.000000000
    - 2021-12-31T18:00:00.000000000
- Out-Sample Time Stamps (used to test inference of Aurora 0.25 Fine-Tuned)
    - 2022-01-01T00:00:00.000000000
    - 2022-12-31T18:00:00.000000000

For more information, refer to docs/hres-t0-source-notes.md

Note: The static variables (land-sea mask, soil type and surface geopotential) required
to form a complete Batch object, are obtained from a separate HuggingFace repository instead
of the cloud storage.

The static variables are downloaded once and cached.

Google Cloud Link: https://console.cloud.google.com/storage/browser/weatherbench2;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false&authuser=1
Last accessed: 07/08/2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import gcsfs
import numpy as np
import pandas as pd
import torch
import zarr
from aurora import Batch, Metadata

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec, validate_input_times

__all__ = ["HresT0Source"]

_WEATHER_DTYPE = torch.float32
OUT_SAMPLE_START = pd.Timestamp("2022-01-01 00:00:00")
OUT_SAMPLE_END = pd.Timestamp("2023-01-01 00:00:00")

# Aurora short name -> WeatherBench2 HRES-T0 long name. See docs/hres-t0-source-notes.md
# for the full variable table. Static vars are deliberately excluded here: they are
# sourced from Microsoft's official static pickle (static_vars.py), not from the WB2
# zarr store, and already use Aurora's short names hence no renaming/remapping needed

# Maps Aurora Batch contract name to its respective hres_t0 name
_SURF_NAME_MAP: dict[str, str] = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
_ATMOS_NAME_MAP: dict[str, str] = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


def _assert_name_maps_cover_spec(spec: ModelSpec) -> None:
    """Ensure every surf/atmos variable in ``spec`` has a WB2 name-map entry."""
    groups: tuple[tuple[str, tuple[str, ...], dict[str, str]], ...] = (
        ("surf_vars", spec.surf_vars, _SURF_NAME_MAP),
        ("atmos_vars", spec.atmos_vars, _ATMOS_NAME_MAP),
    )
    for group_name, required, name_map in groups:
        missing = [key for key in required if key not in name_map]
        if len(missing) > 0:
            missing_str = ", ".join(missing)
            msg = f"HRES-T0 {group_name}: missing WB2 name-map entries for {missing_str}"
            raise AssertionError(msg)


_assert_name_maps_cover_spec(AURORA_PRETRAINED_SPEC)


def open_connection_to_gcs(gcs_store_link: str) -> zarr.Group:
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(gcs_store_link)
    result = zarr.open(store, mode="r")
    assert isinstance(result, zarr.Group), (
        f"expected a zarr.Group at {gcs_store_link}, got {type(result)}"
    )
    return result


class InvalidInitTimeError(Exception):
    """Raised when an input init. time is considered invalid for data source"""


class InvalidPressureLevels(Exception):
    """Raised when an input tuple of pressure levels does not match the model spec"""


@dataclass(frozen=True)
class HresT0Source:
    """
    Data seam for the HRES_T0 dataset
    Returns Batch object of dimension batch_size = 1 only
    """

    ZARR_DATA: zarr.Group
    STATIC_VARS: dict
    batch_size: int = 1

    def load(self, init_time: datetime, spec: ModelSpec) -> Batch:
        """Construct a contract-shaped, single dimension Batch object from the HRES_T0 source"""
        hours = spec.input_timestep_hours
        prev_time = init_time - timedelta(hours=hours)
        validate_input_times(t0=prev_time, t1=init_time, hours=hours)

        # timestamp validity
        self._check_timestamp_valid(
            single_timestamp=prev_time,
            allowed_hours=(18, 6),
            tolerance=hours,
            zarr_data=self.ZARR_DATA,
        )
        self._check_timestamp_valid(
            single_timestamp=init_time,
            allowed_hours=(0, 12),
            tolerance=0,
            zarr_data=self.ZARR_DATA
        )
        timestamp_indices = self._get_timestamp_indices([prev_time, init_time], self.ZARR_DATA)

        # pressure level validity
        pressure_levels = tuple(int(x) for x in self.ZARR_DATA["level"][:])
        self._check_pressure_levels_valid(pressure_levels, spec.atmos_levels)

        # Fetch exactly the variables `spec` requires - not a hardcoded set - so a
        # different ModelSpec (e.g. a future finetuned/1.5-ENS entry) changes what
        # gets fetched without touching this method.
        surf_vars = {
            key: _load_surf_var(self.ZARR_DATA, _SURF_NAME_MAP[key], timestamp_indices)
            for key in spec.surf_vars
        }
        static_vars = {key: _load_static_var(self.STATIC_VARS, key) for key in spec.static_vars}
        atmos_vars = {
            key: _load_atmos_var(self.ZARR_DATA, _ATMOS_NAME_MAP[key], timestamp_indices)
            for key in spec.atmos_vars
        }

        metadata = Metadata(
            lat=torch.tensor(
                # (H,) is 1D, so axis=-2 is out of bounds here - unlike the (B,T,H,W)/
                # (B,T,L,H,W)/(H,W) tensors above, H is axis 0, not second-to-last.
                _flip_to_descending(self.ZARR_DATA["latitude"][:], axis=0),
                dtype=torch.float32,
            ),
            lon=torch.tensor(self.ZARR_DATA["longitude"][:], dtype=torch.float32),
            time=(init_time,),
            atmos_levels=pressure_levels,
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )

    @staticmethod
    def _check_timestamp_valid(
        single_timestamp: datetime,
        allowed_hours: tuple[int, int],
        tolerance: int,
        zarr_data: zarr.Group,
    ) -> None:

        times = pd.to_datetime(zarr_data["time"][:], unit="h", origin="2016-01-01")
        adjusted_out_sample_start = OUT_SAMPLE_START - timedelta(hours=tolerance)

        is_valid_hour = single_timestamp.hour in allowed_hours
        is_in_out_sample_period = (single_timestamp >= adjusted_out_sample_start) & (
            single_timestamp < OUT_SAMPLE_END
        )

        if single_timestamp not in times:
            raise InvalidInitTimeError(f"{single_timestamp} is not found within the HRES_T0 source")
        elif not is_valid_hour:
            raise InvalidInitTimeError(
                f"{single_timestamp} can only take hour values of"
                f"{allowed_hours[0]}UTC or {allowed_hours[1]}UTC"
            )
        elif not is_in_out_sample_period:
            raise InvalidInitTimeError(
                f"{single_timestamp} must lie within the out-sample period "
                f"{adjusted_out_sample_start} to {OUT_SAMPLE_END}(exclusive)"
            )
        else:
            return

    @staticmethod
    def _get_timestamp_indices(time_stamps: list[datetime], zarr_data: zarr.Group) -> np.ndarray:
        times = pd.to_datetime(zarr_data["time"][:], unit="h", origin="2016-01-01")
        return np.where(times.isin(time_stamps))[0]

    @staticmethod
    def _check_pressure_levels_valid(
        pressure_levels: tuple[int, ...], reference_pressure_levels: tuple[int, ...]
    ) -> None:
        if pressure_levels != reference_pressure_levels:
            raise InvalidPressureLevels(
                f"Pressure levels {pressure_levels} do not"
                f"match the model {reference_pressure_levels} in elements and/or order"
            )
        else:
            return


def _load_surf_var(
    zarr_data: zarr.Group, wb2_name: str, timestep_indices: np.ndarray
) -> torch.Tensor:
    """Select the two input timesteps of a ``(time, lat, lon)`` surface variable.

    Adds the batch axis and flips the H (latitude) axis to descending, matching the
    ``metadata.lat`` flip applied below. Returns a ``(1, 2, H, W)`` ``float32`` tensor.
    """
    array = zarr_data[wb2_name][timestep_indices, :][None]
    array = _flip_to_descending(array, axis=-2)  # (B, T, H, W) -> H is second-to-last
    return torch.tensor(array, dtype=_WEATHER_DTYPE)


def _load_atmos_var(
    zarr_data: zarr.Group, wb2_name: str, timestep_indices: np.ndarray
) -> torch.Tensor:
    """Select the two input timesteps of a ``(time, level, lat, lon)`` atmospheric variable.

    Adds the batch axis and flips the H (latitude) axis to descending. Returns a
    ``(1, 2, L, H, W)`` ``float32`` tensor.
    """
    array = zarr_data[wb2_name][timestep_indices, :, :][None]
    array = _flip_to_descending(array, axis=-2)  # (B, T, L, H, W) -> H is second-to-last
    return torch.tensor(array, dtype=_WEATHER_DTYPE)


def _load_static_var(static_vars: dict, key: str) -> torch.Tensor:
    """Flip a cached ``(H, W)`` static variable's H axis to descending, cast to float32."""
    array = _flip_to_descending(static_vars[key][:], axis=-2)  # (H, W) -> H is second-to-last
    return torch.tensor(array, dtype=_WEATHER_DTYPE)


def _flip_to_descending(array: np.ndarray, *, axis: int) -> np.ndarray:
    """Reverse ``array`` along ``axis`` and return a contiguous copy.

    Converts HRES-T0's native ascending latitude into Aurora's required descending
    order. Every tensor that shares this coordinate (surf/atmos/static data, and the
    ``lat`` coordinate itself) must flip its *matching* axis - see
    docs/batch-contract.md's lockstep flip rule.

    The ``.copy()`` is required, not cosmetic: ``np.flip`` returns a negative-stride
    view, and ``torch.tensor()`` cannot consume negative strides.
    """
    return np.flip(array, axis=axis).copy()

    
