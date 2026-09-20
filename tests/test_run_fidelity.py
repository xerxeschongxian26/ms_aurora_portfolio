"""Q2 fidelity harness: --max-steps cap and archive pairing (no Aurora weights)."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import make_valid_batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.evaluation.baselines import (
    FIELDS_HEADLINE,
    init_forecast_dir,
    lead_zarr_path,
    select_persist_fields,
    write_lead_forecast,
)
from aurora_inference.evaluation.grids import batch_to_dataset

_REPO = Path(__file__).resolve().parents[1]
_FIDELITY_CLI = _REPO / "scripts" / "run_fidelity.py"
_INIT_TIME = datetime(2022, 1, 1, 12, 0)
_STEP_HOURS = AURORA_PRETRAINED_SPEC.input_timestep_hours


def _load_fidelity_cli() -> Any:
    spec = importlib.util.spec_from_file_location("run_fidelity_cli", _FIDELITY_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fidelity_cli() -> Any:
    return _load_fidelity_cli()


def test_capped_rollout_steps_truncates_without_extending(fidelity_cli: Any) -> None:
    assert fidelity_cli._capped_rollout_steps(40, None) == 40
    assert fidelity_cli._capped_rollout_steps(40, 10) == 10
    assert fidelity_cli._capped_rollout_steps(40, 40) == 40
    assert fidelity_cli._capped_rollout_steps(40, 50) == 40


def test_parse_args_max_steps(fidelity_cli: Any) -> None:
    assert fidelity_cli._parse_args(["--max-steps", "10"]).max_steps == 10
    assert fidelity_cli._parse_args([]).max_steps is None


def test_main_rejects_non_positive_max_steps(fidelity_cli: Any) -> None:
    assert fidelity_cli.main(["--max-steps", "0", "--baseline-dir", "x"]) == 2


def test_archive_pairing_leaves_unused_leads_unread(fidelity_cli: Any, tmp_path: Path) -> None:
    persist_init_id = 1
    pred = make_valid_batch(height=3, width=4, init_time=_INIT_TIME)
    headline = select_persist_fields(batch_to_dataset(pred), FIELDS_HEADLINE)
    init_dir = init_forecast_dir(tmp_path, persist_init_id)
    for lead_hours in (6, 12, 18, 24):
        write_lead_forecast(headline, lead_zarr_path(init_dir, lead_hours))

    rows = fidelity_cli._score_init_vs_baseline_archive(
        [pred, pred],
        baseline_dir=tmp_path,
        persist_init_id=persist_init_id,
        init_time=_INIT_TIME,
        step_hours=_STEP_HOURS,
    )
    scored = {int(row["lead_hours"]) for row in rows}
    assert scored == {6, 12}
    assert lead_zarr_path(init_dir, 18).is_dir()
    assert lead_zarr_path(init_dir, 24).is_dir()
