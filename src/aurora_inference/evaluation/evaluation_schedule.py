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
    "campaign_n_inits",
    "load_eval_schedule_config",
]

ALLOWED_INIT_HOURS: frozenset[int] = frozenset({0, 12})
EVAL_YEAR = 2022


class SpliceCoverageError(Exception):
    """Raised when a rollout campaign needs times that are not in the splice."""


class EvalScheduleConfig(BaseModel):
    """Knobs shared by splice.toml (write) and rollout.toml (read / coverage).

    Two mutually exclusive ways to name inits:

    * Grid: ``first_init``, ``n_inits``, ``init_stride_hours`` (toy / packed).
    * Explicit: ``inits`` (year-spread that must skip known HRES-T0 holes).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    splice_path: Path
    n_rollout_steps: int = Field(ge=1)
    input_timestep_hours: int = Field(
        default=AURORA_PRETRAINED_SPEC.input_timestep_hours,
        ge=1,
    )
    first_init: datetime | None = None
    n_inits: int | None = Field(default=None, ge=1)
    init_stride_hours: int | None = Field(default=None, gt=0)
    inits: tuple[datetime, ...] | None = None

    @field_validator("splice_path")
    @classmethod
    def _splice_path_non_empty(cls, value: Path) -> Path:
        if not str(value).strip():
            msg = "splice_path must be a non-empty path"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _one_init_schedule_mode(self) -> EvalScheduleConfig:
        has_inits = self.inits is not None
        grid_fields = (self.first_init, self.n_inits, self.init_stride_hours)
        n_grid = sum(field is not None for field in grid_fields)
        if has_inits and n_grid:
            msg = "inits cannot be combined with first_init, n_inits, or init_stride_hours"
            raise ValueError(msg)
        if has_inits:
            inits = self.inits
            if inits is None or len(inits) == 0:
                msg = "inits must contain at least one timestamp"
                raise ValueError(msg)
            if len(set(inits)) != len(inits):
                msg = "inits must be unique"
                raise ValueError(msg)
            ordered = tuple(sorted(inits))
            if ordered != inits:
                return self.model_copy(update={"inits": ordered})
            return self
        if n_grid != 3:
            msg = "provide inits, or all of first_init, n_inits, and init_stride_hours"
            raise ValueError(msg)
        assert self.init_stride_hours is not None
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


def _init_times(config: EvalScheduleConfig) -> pd.DatetimeIndex:
    """Return campaign init timestamps from ``inits`` or the regular grid."""
    if config.inits is not None:
        return pd.DatetimeIndex(config.inits)
    assert config.first_init is not None
    assert config.n_inits is not None
    assert config.init_stride_hours is not None
    return pd.date_range(
        config.first_init,
        periods=config.n_inits,
        freq=pd.Timedelta(hours=config.init_stride_hours),
    )


def campaign_n_inits(config: EvalScheduleConfig) -> int:
    """Number of unique 00/12 eval inits named by this config."""
    if config.inits is not None:
        return len(config.inits)
    assert config.n_inits is not None
    return config.n_inits


def build_eval_schedule(config: EvalScheduleConfig) -> EvalSchedule:
    """Derive init pairs and the union of required 6-hourly HRES-T0 times."""
    step = pd.Timedelta(hours=config.input_timestep_hours)
    inits = _init_times(config)
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
    splice_config: dict[str, object] = {
        "n_inits": campaign_n_inits(cfg),
        "n_rollout_steps": cfg.n_rollout_steps,
    }
    if cfg.inits is not None:
        splice_config["inits"] = [stamp.isoformat() for stamp in cfg.inits]
        splice_config["first_init"] = cfg.inits[0].isoformat()
    else:
        assert cfg.first_init is not None
        assert cfg.init_stride_hours is not None
        splice_config["first_init"] = cfg.first_init.isoformat()
        splice_config["init_stride_hours"] = cfg.init_stride_hours
    return {
        "aurora_inference.splice_config": splice_config,
        "aurora_inference.time_start": times[0].isoformat(),
        "aurora_inference.time_end": times[-1].isoformat(),
        "aurora_inference.n_times": int(len(times)),
    }
