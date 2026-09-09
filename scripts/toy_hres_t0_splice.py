"""Toy Stage 2 splice: download from splice.toml, then coverage-check rollout.toml.

Does not run Aurora. The rollout path stops after asserting that the run's
needed times are a subset of the local splice ``time`` index.

Progress goes to stdout and ``outputs/<tag>/toy_hres_t0_splice.log``.

From the repo root (needs network for GCS):

    uv run --extra forecast python scripts/toy_hres_t0_splice.py --tag toy-splice
    uv run --extra forecast python scripts/toy_hres_t0_splice.py --skip-download
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import zarr

from aurora_inference.data.hres_t0 import _read_zarr_time_index
from aurora_inference.data.hres_t0_splice import copy_hres_t0_splice
from aurora_inference.evaluation.evaluation_schedule import (
    SpliceCoverageError,
    assert_times_available,
    build_eval_schedule,
    campaign_n_inits,
    load_eval_schedule_config,
)
from aurora_inference.logging import (
    configure_run_logging,
    format_elapsed,
    normalize_run_tag,
    run_artifact_dir,
)

_LOG = logging.getLogger(__name__)

_DEFAULT_SPLICE = Path("configs/hres_t0_toy_splice.toml")
_DEFAULT_ROLLOUT = Path("configs/hres_t0_toy_rollout.toml")
_OUTPUT_DIR = Path("outputs")
_LOG_NAME = "toy_hres_t0_splice.log"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splice-config",
        type=Path,
        default=_DEFAULT_SPLICE,
        help="TOML that sizes the local HRES-T0 splice (default: toy 2 inits × 3 steps)",
    )
    parser.add_argument(
        "--rollout-config",
        type=Path,
        default=_DEFAULT_ROLLOUT,
        help="TOML for the rollout campaign; only coverage is checked",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="skip GCS copy; only check rollout.toml against the existing splice",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes the progress log under outputs/<tag>/",
    )
    return parser.parse_args(argv)


def _download_splice(splice_config_path: Path) -> Path:
    config = load_eval_schedule_config(splice_config_path)
    eval_schedule = build_eval_schedule(config)
    _LOG.info(
        "downloading %s unique times (%s → %s) to %s",
        len(eval_schedule.needed_times),
        eval_schedule.needed_times[0],
        eval_schedule.needed_times[-1],
        config.splice_path,
    )
    download_t0 = time.perf_counter()
    out_path = copy_hres_t0_splice(eval_schedule)
    _LOG.info(
        "download wall time: %s (includes GCS open and dest reset)",
        format_elapsed(time.perf_counter() - download_t0),
    )
    _LOG.info("wrote splice %s", out_path)
    return out_path


def _check_rollout_coverage(rollout_config_path: Path) -> None:
    config = load_eval_schedule_config(rollout_config_path)
    eval_schedule = build_eval_schedule(config)
    splice_path = config.splice_path
    if not splice_path.is_absolute():
        splice_path = Path.cwd() / splice_path
    if not splice_path.exists():
        msg = f"splice not found at {splice_path}"
        raise FileNotFoundError(msg)

    group = zarr.open_group(splice_path, mode="r")
    if not isinstance(group, zarr.Group):
        msg = f"expected zarr.Group at {splice_path}, got {type(group)}"
        raise TypeError(msg)
    available = _read_zarr_time_index(group)
    assert_times_available(eval_schedule.needed_times, available)
    _LOG.info(
        "rollout coverage ok: %s inits × %s steps (%s unique times) ⊆ %s (%s times)",
        campaign_n_inits(config),
        config.n_rollout_steps,
        len(eval_schedule.needed_times),
        splice_path,
        len(available),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tag = normalize_run_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    output_dir = run_artifact_dir(_OUTPUT_DIR, tag)
    log_path = output_dir / _LOG_NAME
    configure_run_logging(log_path, tag=tag)
    _LOG.info("Splice HRES-T0; log file: %s", log_path)

    if not args.skip_download:
        _download_splice(args.splice_config)
    try:
        _check_rollout_coverage(args.rollout_config)
    except SpliceCoverageError as exc:
        _LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
