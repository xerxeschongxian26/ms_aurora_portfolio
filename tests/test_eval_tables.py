"""RMSE table write / resume helpers (no Aurora weights)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aurora_inference.evaluation.tables import (
    MEAN_BY_LEAD_NAME,
    PER_INIT_NAME,
    load_rmse_rows,
    write_rmse_tables,
)


def _row(
    *,
    init_id: int,
    lead_hours: int,
    variable: str,
    rmse: float,
    level: int | None = None,
) -> dict[str, object]:
    return {
        "init_id": init_id,
        "init_time": "2022-01-01T12:00:00",
        "lead_hours": lead_hours,
        "variable": variable,
        "level": level,
        "rmse": rmse,
    }


def test_write_rmse_tables_means_over_inits(tmp_path: Path) -> None:
    rows = [
        _row(init_id=1, lead_hours=6, variable="2t", rmse=2.0),
        _row(init_id=2, lead_hours=6, variable="2t", rmse=4.0),
        _row(init_id=1, lead_hours=6, variable="z", rmse=3.0, level=500),
        _row(init_id=2, lead_hours=6, variable="z", rmse=9.0, level=500),
    ]
    out_path = write_rmse_tables(rows, tmp_path)
    assert out_path == tmp_path / MEAN_BY_LEAD_NAME
    mean = pd.read_csv(out_path)
    two_t = mean.loc[mean["variable"] == "2t", "mean_rmse"]
    z500 = mean.loc[(mean["variable"] == "z") & (mean["level"] == 500), "mean_rmse"]
    assert float(two_t.iloc[0]) == 3.0
    assert float(z500.iloc[0]) == 6.0
    assert (tmp_path / PER_INIT_NAME).exists()


def test_load_rmse_rows_roundtrip(tmp_path: Path) -> None:
    assert load_rmse_rows(tmp_path) == []
    write_rmse_tables([_row(init_id=7, lead_hours=12, variable="msl", rmse=1.5)], tmp_path)
    loaded = load_rmse_rows(tmp_path)
    assert len(loaded) == 1
    assert int(loaded[0]["init_id"]) == 7
    assert float(loaded[0]["rmse"]) == 1.5


def test_write_rmse_tables_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no RMSE rows"):
        write_rmse_tables([], tmp_path)
