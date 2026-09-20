"""Write RMSE campaign tables (per-init long form and mean by lead)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

__all__ = ["load_rmse_rows", "write_rmse_tables"]

PER_INIT_NAME = "rmse_by_init.csv"
MEAN_BY_LEAD_NAME = "rmse_by_lead.csv"


def load_rmse_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Load checkpointed per-init RMSE rows, or an empty list if none exist."""
    path = output_dir / PER_INIT_NAME
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def write_rmse_tables(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write ``rmse_by_init.csv`` and the mean-over-inits ``rmse_by_lead.csv``."""
    if not rows:
        msg = "no RMSE rows to write"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / PER_INIT_NAME, index=False)
    grouped = frame.groupby(["lead_hours", "variable", "level"], dropna=False, as_index=False).agg(
        mean_rmse=("rmse", "mean")
    )
    grouped = grouped.sort_values(["lead_hours", "variable", "level"], na_position="first")
    out_path = output_dir / MEAN_BY_LEAD_NAME
    grouped.to_csv(out_path, index=False)
    return out_path
