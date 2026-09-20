"""CPU streamed non-finite checker for a WB2 HRES-T0 zarr (GCS or local).

Does not load Aurora weights and does not call ``eval_rmse.py``.

Examples (from the repo root)::

    uv run --extra forecast python scripts/check_hres_t0_finite.py \\
        --store data/splice_hres_t0_toy_holes.zarr \\
        --start 2022-01-01T06 --end 2022-01-02T18 \\
        --output-dir outputs --output-name toy-holes.csv

    uv run --extra forecast python scripts/check_hres_t0_finite.py --out-sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from aurora_inference.config import GCS_STORE_LINK
from aurora_inference.data.hres_t0_finite import (
    DEFAULT_GRID,
    FiniteCheckConfigError,
    GridSpec,
    resolve_output_paths,
    run_finite_check,
)
from aurora_inference.logging import configure_run_logging

_LOG = logging.getLogger(__name__)

_GCSFS_SHUTDOWN = ("event loop is closed", "cannot schedule new futures")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        default=GCS_STORE_LINK,
        help="local .zarr path or gs:// URI (default: WB2 HRES-T0 GCS store)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="inclusive window start (ignored when --out-sample)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="inclusive window end (ignored when --out-sample)",
    )
    parser.add_argument(
        "--out-sample",
        action="store_true",
        help="scan 2022-01-01T00 .. 2022-12-31T18 and ignore --start/--end",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for the CSV catalogue and log file",
    )
    parser.add_argument(
        "--output-name",
        default="hres-t0-nonfinite.csv",
        help="catalogue file name (always written as .csv)",
    )
    parser.add_argument(
        "--all-vars",
        action="store_true",
        help="scan every weather array in the group, not only the Aurora-9 maps",
    )
    parser.add_argument(
        "--n-lat",
        type=int,
        default=DEFAULT_GRID.n_lat,
        help="expected latitude size (default: 721)",
    )
    parser.add_argument(
        "--n-lon",
        type=int,
        default=DEFAULT_GRID.n_lon,
        help="expected longitude size (default: 1440)",
    )
    parser.add_argument(
        "--n-level",
        type=int,
        default=DEFAULT_GRID.n_level,
        help="expected pressure-level size (default: 13)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="attempts per array fetch on retryable GCS/I/O errors",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip times listed in the sibling .progress file",
    )
    return parser.parse_args(argv)


def _is_gcsfs_shutdown(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return any(hint in text for hint in _GCSFS_SHUTDOWN)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    _, log_path = resolve_output_paths(output_dir, args.output_name)
    configure_run_logging(log_path)

    start = None if args.start is None else pd.Timestamp(args.start)
    end = None if args.end is None else pd.Timestamp(args.end)
    grid = GridSpec(n_lat=args.n_lat, n_lon=args.n_lon, n_level=args.n_level)
    try:
        result = run_finite_check(
            store=str(args.store),
            output_dir=output_dir,
            output_name=args.output_name,
            grid=grid,
            start=start,
            end=end,
            out_sample=args.out_sample,
            all_vars=args.all_vars,
            max_retries=args.max_retries,
            resume=args.resume,
        )
    except FiniteCheckConfigError as exc:
        _LOG.error("Configuration Invalid | %s", exc)
        return 2

    _LOG.info(
        "done: %s hole rows, %s/%s times with invalid values, catalogue=%s",
        result.n_hole_rows,
        result.n_times_with_holes,
        result.n_times,
        result.catalogue_path,
    )
    return 1 if result.n_hole_rows > 0 else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        if _is_gcsfs_shutdown(exc):
            _LOG.warning("ignoring gcsfs shutdown error after run: %s", exc)
            raise SystemExit(0) from exc
        raise
