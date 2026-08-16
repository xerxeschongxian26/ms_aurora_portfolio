"""Real forecast on a GPU: HRES-T0 → Aurora() rollout → one 2t PNG per step.

Aurora() possesses forecast skill. Recall that Aurora(), is AuroraPretrained that
has been fine-tuned on the HRES-T0 dataset.

Requires the ``viz`` extra (and ``dev`` for zarr/gcsfs)::

    uv sync --extra dev --extra viz
    python scripts/real_forecast.py --steps 4
"""

from __future__ import annotations

import argparse
import logging
import platform
import resource
import sys
import time

import torch

from datetime import datetime
from pathlib import Path

from plot_forecast import plot_batch_2t

from aurora_inference.config import GCS_STORE_LINK
from aurora_inference.contract import AURORA_PRETRAINED_SPEC, validate_batch
from aurora_inference.data.hres_t0 import HresT0Source, open_connection_to_gcs
from aurora_inference.data.static_vars import get_hres_t0_static
from aurora_inference.inference.forward import run_forecast
from aurora_inference.model.loader import load_model

_LOG = logging.getLogger(__name__)

_MODEL_NAME =  "aurora-finetuned"
_DEVICE = "cuda"
_DEFAULT_STEPS = 4
_INIT_TIME = datetime(2022, 6, 15, 12, 0)
_OUTPUT_DIR = Path("outputs")

_SKILL_BANNER = "=== FORECAST SKILL === Aurora performs a real forecast."


def _peak_rss_bytes() -> int:
    """Return peak resident set size (RSS) in bytes (platform-normalized).

    ``ru_maxrss`` units differ by OS: macOS reports bytes; Linux reports kilobytes.
    This helper normalises the value to bytes.

    Notes:
        - Statistics are per-process. External apps (e.g. a RAM-heavy browser) do not affect
          the recorded peak, though heavy system load may slow ``load_seconds`` /
          ``forward_seconds``.
        - Peak RSS is a high-water mark since this Python process started (imports,
          model load, forward), not current memory at call time.
    """
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if platform.system() == "Darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_STEPS,
        help=f"rollout steps (default {_DEFAULT_STEPS}; WP4 acceptance uses 4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    _LOG.info(_SKILL_BANNER)

    # SystemExit if cuda is unavailable
    if not torch.cuda.is_available():
        _LOG.error("_DEVICE = %s but CUDA device is unavailable", _DEVICE)
        return 1

    # Create a HresT0 Batch object
    zarr_data = open_connection_to_gcs(gcs_store_link=GCS_STORE_LINK)
    static_vars = get_hres_t0_static()
    source = HresT0Source(ZARR_DATA=zarr_data, STATIC_VARS=static_vars)
    batch = source.load(_INIT_TIME, AURORA_PRETRAINED_SPEC)
    validate_batch(batch, AURORA_PRETRAINED_SPEC)
    batch = batch.to(_DEVICE)

    _LOG.info(
        "input: surf 2t=%s atmos t=%s, init=%s device=%s",
        tuple(batch.surf_vars["2t"].shape),
        tuple(batch.atmos_vars["t"].shape),
        _INIT_TIME.isoformat(),
        _DEVICE,
    )

    load_started = time.perf_counter()
    model = load_model(model_name=_MODEL_NAME, device=_DEVICE)
    _LOG.info("model load wall time: %.2fs", time.perf_counter() - load_started)

    rollout_started = time.perf_counter()
    forecasts = run_forecast(model, batch, steps=args.steps)
    _LOG.info(
        "run_forecast wall time: %.2fs for %d steps",
        time.perf_counter() - rollout_started,
        args.steps,
    )

    for i, pred in enumerate(forecasts, start=1):
        output_path = _OUTPUT_DIR / f"real_forecast_2t_step{i:02d}.png"
        plot_batch_2t(pred, output_path)

    peak_rss_bytes = _peak_rss_bytes()
    _LOG.info("peak RSS: %.2f MiB (%d bytes)", peak_rss_bytes / (1024 * 1024), peak_rss_bytes)
    _LOG.warning(_SKILL_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
