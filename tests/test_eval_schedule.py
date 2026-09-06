from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from aurora_inference.evaluation.evaluation_schedule import (
    EvalScheduleConfig,
    SpliceCoverageError,
    assert_times_available,
    build_eval_schedule,
    load_eval_schedule_config,
)

_REPO = Path(__file__).resolve().parents[1]
_TOY_SPLICE = _REPO / "configs" / "hres_t0_toy_splice.toml"
_TOY_ROLLOUT = _REPO / "configs" / "hres_t0_toy_rollout.toml"
_SPREAD_SPLICE = _REPO / "configs" / "hres_t0_2022_spread_splice.toml"
_SPREAD_ROLLOUT = _REPO / "configs" / "hres_t0_2022_spread_rollout.toml"


def _toy_config(**overrides: object) -> EvalScheduleConfig:
    payload: dict[str, object] = {
        "splice_path": "notebooks/_scratch/splice_hres_t0_toy.zarr",
        "first_init": datetime(2022, 1, 1, 12, 0),
        "n_inits": 2,
        "init_stride_hours": 12,
        "n_rollout_steps": 3,
    }
    payload.update(overrides)
    return EvalScheduleConfig.model_validate(payload)


def test_toy_splice_toml_has_seven_overlapping_times() -> None:
    config = load_eval_schedule_config(_TOY_SPLICE)
    eval_schedule = build_eval_schedule(config)
    assert config.n_inits == 2
    assert config.n_rollout_steps == 3
    assert len(eval_schedule.init_pairs) == 2
    assert len(eval_schedule.needed_times) == 7
    assert eval_schedule.needed_times[0] == pd.Timestamp("2022-01-01T06")
    assert eval_schedule.needed_times[-1] == pd.Timestamp("2022-01-02T18")
    assert eval_schedule.init_pairs[0].init_time == pd.Timestamp("2022-01-01T12")
    assert eval_schedule.init_pairs[1].init_time == pd.Timestamp("2022-01-02T00")
    assert eval_schedule.init_pairs[0].last_rollout == pd.Timestamp("2022-01-02T06")
    assert eval_schedule.init_pairs[1].last_rollout == pd.Timestamp("2022-01-02T18")


def test_toy_rollout_toml_matches_splice_needed_times() -> None:
    splice_eval_schedule = build_eval_schedule(load_eval_schedule_config(_TOY_SPLICE))
    rollout_eval_schedule = build_eval_schedule(load_eval_schedule_config(_TOY_ROLLOUT))
    assert_times_available(rollout_eval_schedule.needed_times, splice_eval_schedule.needed_times)


def test_assert_times_available_raises_on_missing_lead() -> None:
    eval_schedule = build_eval_schedule(_toy_config())
    hole = eval_schedule.needed_times[:-1]
    with pytest.raises(SpliceCoverageError, match="missing"):
        assert_times_available(eval_schedule.needed_times, hole)


def test_build_eval_schedule_rejects_non_2022_init() -> None:
    with pytest.raises(ValueError, match="not in 2022"):
        build_eval_schedule(_toy_config(first_init=datetime(2021, 1, 1, 12, 0)))


def test_build_eval_schedule_rejects_06_utc_init() -> None:
    with pytest.raises(ValueError, match="00UTC"):
        build_eval_schedule(_toy_config(first_init=datetime(2022, 1, 1, 6, 0)))


def test_eval_schedule_config_rejects_n_inits_zero() -> None:
    with pytest.raises(ValueError):
        _toy_config(n_inits=0)


def test_spread_toml_covers_four_quarters() -> None:
    config = load_eval_schedule_config(_SPREAD_SPLICE)
    eval_schedule = build_eval_schedule(config)
    inits = [pair.init_time for pair in eval_schedule.init_pairs]
    by_quarter = pd.Series(inits).dt.quarter.value_counts().sort_index()
    assert config.n_inits == 30
    assert config.n_rollout_steps == 40
    assert config.init_stride_hours == 300
    assert list(by_quarter) == [8, 7, 7, 8]
    assert {ts.hour for ts in inits} == {0, 12}
    assert inits[0] == pd.Timestamp("2022-01-01T12")
    assert inits[-1] == pd.Timestamp("2022-12-30T00")
    assert eval_schedule.needed_times[0] == pd.Timestamp("2022-01-01T06")
    assert eval_schedule.needed_times[-1] == pd.Timestamp("2023-01-09T00")
    assert len(eval_schedule.needed_times) == 1260


def test_spread_rollout_toml_matches_splice_needed_times() -> None:
    splice_eval_schedule = build_eval_schedule(load_eval_schedule_config(_SPREAD_SPLICE))
    rollout_eval_schedule = build_eval_schedule(load_eval_schedule_config(_SPREAD_ROLLOUT))
    assert_times_available(rollout_eval_schedule.needed_times, splice_eval_schedule.needed_times)
