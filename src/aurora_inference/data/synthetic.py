"""Synthetic batch source for CI and plumbing tests.

THIS IS A TEST FIXTURE, NOT A DATA SOURCE.

It produces correctly shaped, correctly oriented, seeded noise that satisfies
the Aurora Batch contract with ZERO network I/O. Forecasts derived from it are
MEANINGLESS — random noise in, noise out. Do not use it on any evaluation path
and never report skill numbers from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import torch
from aurora import Batch, Metadata

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec, validate_input_times

__all__ = ["SyntheticSource"]

_INPUT_TIME_STEPS = 2
_WEATHER_DTYPE = torch.float32

# Physically plausible centers/scales for known Aurora short names.
# Every key in AURORA_PRETRAINED_SPEC must appear here — no silent fallbacks.
# Values are decorative — not real earth states.
_SURF_RANGES: dict[str, tuple[float, float]] = {
    "2t": (273.0, 20.0),
    "10u": (0.0, 10.0),
    "10v": (0.0, 10.0),
    "msl": (101_325.0, 1_000.0),
}
_STATIC_RANGES: dict[str, tuple[float, float]] = {
    "lsm": (0.5, 0.5),
    "slt": (3.0, 2.0),
    "z": (0.0, 5_000.0),
}
_ATMOS_RANGES: dict[str, tuple[float, float]] = {
    "t": (250.0, 30.0),
    "u": (0.0, 20.0),
    "v": (0.0, 20.0),
    "q": (0.005, 0.003),
    "z": (50_000.0, 20_000.0),
}


def _assert_range_tables_cover_spec(spec: ModelSpec) -> None:
    """Ensure every variable in ``spec`` has a defined sampling range."""
    groups: tuple[tuple[str, tuple[str, ...], dict[str, tuple[float, float]]], ...] = (
        ("surf_vars", spec.surf_vars, _SURF_RANGES),
        ("static_vars", spec.static_vars, _STATIC_RANGES),
        ("atmos_vars", spec.atmos_vars, _ATMOS_RANGES),
    )
    for group_name, required, ranges in groups:
        missing = [key for key in required if key not in ranges]
        if len(missing) > 0:
            missing_str = ", ".join(missing)
            msg = f"synthetic {group_name}: missing range entries for {missing_str}"
            raise AssertionError(msg)


_assert_range_tables_cover_spec(AURORA_PRETRAINED_SPEC)


@dataclass(frozen=True)
class SyntheticSource:
    """Seeded synthetic ``BatchSource`` for offline tests.

    Default grid is small (``H=32``, ``W=64``) for fast CI. Full Aurora global
    resolution ``(721, 1440)`` is allowed when explicitly requested.
    """

    height: int = 32
    width: int = 64
    seed: int = 42
    batch_size: int = 1

    def load(self, init_time: datetime, spec: ModelSpec) -> Batch:
        """Synthesize a contract-shaped batch whose ``metadata.time`` is ``init_time`` (t1)."""
        hours = spec.input_timestep_hours
        t0 = init_time - timedelta(hours=hours)
        validate_input_times(t0=t0, t1=init_time, hours=hours)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # Build orientation correctly at construction — never flip coords alone later.
        lat = torch.linspace(90.0, -90.0, self.height, dtype=torch.float32)
        lon = torch.linspace(
            0.0,
            360.0 - (360.0 / self.width),
            self.width,
            dtype=torch.float32,
        )

        level_count = len(spec.atmos_levels)
        surf_shape = (self.batch_size, _INPUT_TIME_STEPS, self.height, self.width)
        atmos_shape = (
            self.batch_size,
            _INPUT_TIME_STEPS,
            level_count,
            self.height,
            self.width,
        )
        static_shape = (self.height, self.width)

        surf_vars = {
            key: self._sample_variable(surf_shape, *_SURF_RANGES[key], generator)
            for key in spec.surf_vars
        }
        static_vars = {
            key: self._sample_variable(static_shape, *_STATIC_RANGES[key], generator)
            for key in spec.static_vars
        }
        atmos_vars = {
            key: self._sample_variable(atmos_shape, *_ATMOS_RANGES[key], generator)
            for key in spec.atmos_vars
        }

        metadata = Metadata(
            lat=lat,
            lon=lon,
            time=(init_time,) * self.batch_size,
            atmos_levels=spec.atmos_levels,
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )

    @staticmethod
    def _sample_variable(
        shape: tuple[int, ...],
        center: float,
        scale: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Draw ``center + scale * N(0, 1)`` as float32."""
        return (center + scale * torch.randn(shape, generator=generator, dtype=_WEATHER_DTYPE)).to(
            dtype=_WEATHER_DTYPE
        )
