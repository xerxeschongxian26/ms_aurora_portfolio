"""WP2.3: re-score a few archive leads with the current FP64 MSE.

Run at the **start** of the WP3 GPU session, before any precision variant.
Compares new RMSE(archive, HRES-T0) against the Stage 2 ``rmse_by_init.csv``
written next to the headline archive. Not a laptop job: the archive and splice
live on NFS. Do not substitute by scoring a forecast against itself.

Example (on the A100 box, NFS attached)::

    uv run --extra forecast python scripts/rescore_baseline_leads.py \\
        --baseline-dir /home/ubuntu/xerxes-nfs/aurora-baselines/wp5b-baselines-n30-headline \\
        --splice-path /home/ubuntu/xerxes-nfs/splice_hres_t0_2022_full.zarr \\
        --init-id 1 --leads 24 120 240
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.data.hres_t0 import HresT0Source, open_local_zarr
from aurora_inference.data.static_vars import get_hres_t0_static
from aurora_inference.evaluation.baselines import (
    FLOOR_LEAD_HOURS,
    init_forecast_dir,
    lead_zarr_path,
    read_lead_forecast,
    score_headline_vs_analysis_rows,
)
from aurora_inference.evaluation.grids import analysis_dataset, as_naive_datetime
from aurora_inference.evaluation.tables import load_rmse_rows, relative_change
from aurora_inference.logging import configure_run_logging, normalize_run_tag, run_artifact_dir

_LOG = logging.getLogger(__name__)

_DEFAULT_BASELINE = Path("/home/ubuntu/xerxes-nfs/aurora-baselines/wp5b-baselines-n30-headline")
_DEFAULT_SPLICE = Path("/home/ubuntu/xerxes-nfs/splice_hres_t0_2022_full.zarr")
_OUTPUT_DIR = Path("outputs/wp2-archive-rescore")
# Session abort heuristic, not a published pass band. Skill RMSE should not move
# by a close-variant's worth (~1e-4 relative) just from the scorer width change.
_MATERIAL_RELATIVE = 1.0e-4


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=_DEFAULT_BASELINE)
    parser.add_argument("--splice-path", type=Path, default=_DEFAULT_SPLICE)
    parser.add_argument(
        "--init-id",
        type=int,
        default=1,
        help="persist init_id (default 1 = 2022-01-01T12)",
    )
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=list(FLOOR_LEAD_HOURS),
        help="lead hours to re-score (default: 24 120 240)",
    )
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--max-relative",
        type=float,
        default=_MATERIAL_RELATIVE,
        help="abort if any row's relative change exceeds this (default 1e-4)",
    )
    return parser.parse_args(argv)


def _init_time_for_id(rows: list[dict[str, Any]], init_id: int) -> datetime:
    for row in rows:
        if int(row["init_id"]) == init_id:
            return as_naive_datetime(pd.Timestamp(row["init_time"]))
    msg = f"no Stage 2 RMSE rows for init_id={init_id}"
    raise FileNotFoundError(msg)


def _row_key(row: dict[str, Any]) -> tuple[int, int, str, int | None]:
    level = row["level"]
    if level is None or (isinstance(level, float) and pd.isna(level)):
        level_key: int | None = None
    else:
        level_key = int(level)
    return (int(row["init_id"]), int(row["lead_hours"]), str(row["variable"]), level_key)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tag = normalize_run_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.init_id < 1:
        print(f"init-id must be >= 1 (got {args.init_id})", file=sys.stderr)
        return 2
    if any(lead < 1 for lead in args.leads):
        print("leads must be >= 1", file=sys.stderr)
        return 2

    output_dir = run_artifact_dir(args.output_dir.expanduser(), tag)
    configure_run_logging(output_dir / "rescore_baseline_leads.log", tag=tag)

    baseline_dir = args.baseline_dir.expanduser()
    stage2_rows = load_rmse_rows(baseline_dir)
    if not stage2_rows:
        _LOG.error("no Stage 2 rmse_by_init.csv under %s", baseline_dir)
        return 1

    try:
        init_time = _init_time_for_id(stage2_rows, args.init_id)
    except FileNotFoundError as exc:
        _LOG.error("%s", exc)
        return 1

    source = HresT0Source(
        ZARR_DATA=open_local_zarr(args.splice_path.expanduser()),
        STATIC_VARS=get_hres_t0_static(),
    )
    init_dir = init_forecast_dir(baseline_dir, args.init_id)
    stage2_by_key = {_row_key(row): float(row["rmse"]) for row in stage2_rows}
    comparisons: list[dict[str, Any]] = []
    max_rel = 0.0

    for lead_hours in args.leads:
        valid_time = init_time + timedelta(hours=lead_hours)
        forecast = read_lead_forecast(lead_zarr_path(init_dir, lead_hours))
        analysis = analysis_dataset(source, valid_time, AURORA_PRETRAINED_SPEC)
        new_rows = score_headline_vs_analysis_rows(
            forecast,
            analysis,
            init_id=args.init_id,
            init_time=init_time,
            lead_hours=lead_hours,
        )
        for new_row in new_rows:
            key = _row_key(new_row)
            old = stage2_by_key.get(key)
            if old is None:
                _LOG.error("no Stage 2 row for %s", key)
                return 1
            new = float(new_row["rmse"])
            rel = relative_change(old, new)
            max_rel = max(max_rel, rel)
            comparisons.append(
                {
                    "init_id": args.init_id,
                    "init_time": init_time.isoformat(),
                    "lead_hours": lead_hours,
                    "variable": new_row["variable"],
                    "level": new_row["level"],
                    "stage2_rmse": old,
                    "fp64_rmse": new,
                    "relative_change": rel,
                }
            )
            _LOG.info(
                "init_id=%s lead=%sh %s level=%s stage2=%.6g fp64=%.6g rel=%.3e",
                args.init_id,
                lead_hours,
                new_row["variable"],
                new_row["level"],
                old,
                new,
                rel,
            )

    payload = {
        "baseline_dir": str(baseline_dir),
        "init_id": args.init_id,
        "leads": list(args.leads),
        "max_relative_change": max_rel,
        "max_relative_abort": args.max_relative,
        "n_rows": len(comparisons),
        "rows": comparisons,
    }
    out_path = output_dir / "rescore.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _LOG.info("wrote %s (max relative change %.3e)", out_path, max_rel)

    if max_rel > args.max_relative:
        _LOG.error(
            "WP2.3 abort: max relative change %.3e exceeds %.3e — stop the session",
            max_rel,
            args.max_relative,
        )
        return 1
    _LOG.info("WP2.3 ok: archive re-score stayed within %.3e relative", args.max_relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
