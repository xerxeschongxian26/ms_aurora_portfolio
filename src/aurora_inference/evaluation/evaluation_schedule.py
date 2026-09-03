"""Build overlapping HRES-T0 eval timestamps from a TOML campaign config.

Packed 00/12 inits share 6-hourly times (t−6h, t, and each lead). ``needed_times``
is the union of those stamps, not one independent window per init.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aurora_inference.contract import AURORA_PRETRAINED_SPEC

__all__ = [
    "ALLOWED_INIT_HOURS",
    "EVAL_YEAR",
    "EvalSchedule",
    "EvalScheduleConfig",
    "InitPair",
    "SpliceCoverageError",
    "assert_times_available",
    "build_eval_schedule",
    "campaign_attr_payload",
    "load_eval_schedule_config",
]

ALLOWED_INIT_HOURS: frozenset[int] = frozenset({0, 12})
EVAL_YEAR = 2022


class SpliceCoverageError(Exception):
    """Raised when a rollout campaign needs times that are not in the splice."""


class EvalScheduleConfig(BaseModel):
    """Knobs shared by splice.toml (write) and rollout.toml (read / coverage)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    splice_path: Path
    first_init: datetime
    n_inits: int = Field(ge=1)
    init_stride_hours: int = Field(gt=0)
    n_rollout_steps: int = Field(ge=1)
    input_timestep_hours: int = Field(
        default=AURORA_PRETRAINED_SPEC.input_timestep_hours,
        ge=1,
    )

    @field_validator("splice_path")
    @classmethod
    def _splice_path_non_empty(cls, value: Path) -> Path:
        if not str(value).strip():
            msg = "splice_path must be a non-empty path"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _stride_is_multiple_of_timestep(self) -> EvalScheduleConfig:
        if self.init_stride_hours % self.input_timestep_hours != 0:
            msg = (
                f"init_stride_hours ({self.init_stride_hours}) must be a "
                f"positive multiple of input_timestep_hours ({self.input_timestep_hours})"
            )
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class InitPair:
    """One eval init: Aurora input (t−6h, t) and first/last forecast valid times."""

    init_id: int
    prev_time: pd.Timestamp
    init_time: pd.Timestamp
    first_rollout: pd.Timestamp
    last_rollout: pd.Timestamp
    n_rollout_steps: int


@dataclass(frozen=True)
class EvalSchedule:
    """Eval timestamps derived from :class:`EvalScheduleConfig`."""

    config: EvalScheduleConfig
    init_pairs: tuple[InitPair, ...]
    needed_times: pd.DatetimeIndex


def load_eval_schedule_config(path: Path) -> EvalScheduleConfig:
    """Parse a splice or rollout TOML into :class:`EvalScheduleConfig`."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return EvalScheduleConfig.model_validate(payload)


def build_eval_schedule(config: EvalScheduleConfig) -> EvalSchedule:
    """Derive init pairs and the union of required 6-hourly HRES-T0 times."""
    step = pd.Timedelta(hours=config.input_timestep_hours)
    inits = pd.date_range(
        config.first_init,
        periods=config.n_inits,
        freq=pd.Timedelta(hours=config.init_stride_hours),
    )
    _assert_inits_are_eval_protocol(inits)

    pairs: list[InitPair] = []
    stamps: list[pd.Timestamp] = []
    for init_id, init_time in enumerate(inits, start=1):
        prev_time = init_time - step
        first_rollout = init_time + step
        last_rollout = init_time + config.n_rollout_steps * step
        leads = pd.date_range(first_rollout, last_rollout, freq=step)
        stamps.extend([prev_time, init_time, *leads])
        pairs.append(
            InitPair(
                init_id=init_id,
                prev_time=pd.Timestamp(prev_time),
                init_time=pd.Timestamp(init_time),
                first_rollout=pd.Timestamp(first_rollout),
                last_rollout=pd.Timestamp(last_rollout),
                n_rollout_steps=config.n_rollout_steps,
            )
        )

    needed_times = pd.DatetimeIndex(stamps).unique().sort_values()
    return EvalSchedule(config=config, init_pairs=tuple(pairs), needed_times=needed_times)


def assert_times_available(
    needed: pd.DatetimeIndex,
    available: pd.DatetimeIndex,
) -> None:
    """Require every ``needed`` stamp to appear in ``available`` (set membership)."""
    missing = needed.difference(available)
    if missing.empty:
        return
    preview = ", ".join(str(ts) for ts in missing[:8])
    extra = "" if len(missing) <= 8 else f" … ({len(missing)} missing)"
    msg = f"splice is missing {len(missing)} required time(s): {preview}{extra}"
    raise SpliceCoverageError(msg)


def _assert_inits_are_eval_protocol(inits: pd.DatetimeIndex) -> None:
    """Q1 inits are 2022 00/12 UTC only (prev t−6h may fall on 2021-12-31)."""
    for init in inits:
        ts = pd.Timestamp(init)
        if ts.year != EVAL_YEAR:
            msg = f"eval init {ts} is not in {EVAL_YEAR}"
            raise ValueError(msg)
        if ts.hour not in ALLOWED_INIT_HOURS or ts.minute != 0 or ts.second != 0:
            allowed = ", ".join(f"{hour:02d}UTC" for hour in sorted(ALLOWED_INIT_HOURS))
            msg = f"eval init {ts} must be {allowed} on the hour"
            raise ValueError(msg)


def campaign_attr_payload(eval_schedule: EvalSchedule) -> dict[str, object]:
    """JSON-serializable identification blob for splice zarr group attrs."""
    cfg = eval_schedule.config
    times = eval_schedule.needed_times
    return {
        "aurora_inference.splice_config": {
            "first_init": cfg.first_init.isoformat(),
            "n_inits": cfg.n_inits,
            "init_stride_hours": cfg.init_stride_hours,
            "n_rollout_steps": cfg.n_rollout_steps,
        },
        "aurora_inference.time_start": times[0].isoformat(),
        "aurora_inference.time_end": times[-1].isoformat(),
        "aurora_inference.n_times": int(len(times)),
    }
