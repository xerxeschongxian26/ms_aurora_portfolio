"""CPU toy forward pass
- A single successful forward pass is proof that the data pipeline works

Loads AuroraSmallPretrained with a small synthetic batch, runs one forward step,
and logs shapes / timing / peak RSS (resident set size). This is not a true inference
as it uses AuroraSmallPretrained on synthetic data and has no forecast skill.
"""

from __future__ import annotations

import logging
import platform
import resource
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch
from aurora import Batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, validate_batch
from aurora_inference.data.synthetic import SyntheticSource
from aurora_inference.model.loader import load_model

_LOG = logging.getLogger(__name__)

_GRID_HEIGHT = 32
_GRID_WIDTH = 64
_DEVICE = "cpu"
_INIT_TIME = datetime(2022, 1, 1, 12, 0, tzinfo=UTC)

_NO_SKILL_BANNER_TOP = (
    "=== NO FORECAST SKILL === "
    "AuroraSmallPretrained + SyntheticSource is a plumbing proof only. "
    "Output is meaningless; do not report skill numbers."
)
_NO_SKILL_BANNER_BOTTOM = "=== NO FORECAST SKILL === "


@dataclass(frozen=True)
class ToyForwardResult:
    """Shapes and timings from a single CPU toy forward."""

    input_surf_shape: tuple[int, ...]
    input_atmos_shape: tuple[int, ...]
    output_surf_shape: tuple[int, ...]
    output_atmos_shape: tuple[int, ...]
    load_seconds: float
    forward_seconds: float
    peak_rss_bytes: int


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


def _assert_output_time_dim_is_one(prediction: Batch) -> None:
    """Aurora's single rollout step returns T==1, not the input T==2."""
    for name, tensor in prediction.surf_vars.items():
        if tensor.shape[1] != 1:
            msg = f"output surf_vars[{name!r}] time dim must be 1, got shape {tuple(tensor.shape)}"
            raise AssertionError(msg)
    for name, tensor in prediction.atmos_vars.items():
        if tensor.shape[1] != 1:
            msg = f"output atmos_vars[{name!r}] time dim must be 1, got shape {tuple(tensor.shape)}"
            raise AssertionError(msg)


def run_toy_forward() -> ToyForwardResult:
    """Run the WP5 CPU plumbing proof and return logged metrics."""
    _LOG.warning(_NO_SKILL_BANNER_TOP)

    # Create a synthetic Batch object
    source = SyntheticSource(height=_GRID_HEIGHT, width=_GRID_WIDTH)
    batch = source.load(_INIT_TIME, AURORA_PRETRAINED_SPEC)
    validate_batch(batch, AURORA_PRETRAINED_SPEC)
    batch = batch.to(_DEVICE)

    input_surf_shape = tuple(batch.surf_vars["2t"].shape)
    input_atmos_shape = tuple(batch.atmos_vars["t"].shape)
    _LOG.info(
        "input shapes: surf 2t=%s atmos t=%s grid=%dx%d device=%s",
        input_surf_shape,
        input_atmos_shape,
        _GRID_HEIGHT,
        _GRID_WIDTH,
        _DEVICE,
    )

    # Load model and log time taken
    load_started = time.perf_counter()
    model = load_model(device=_DEVICE)
    load_seconds = time.perf_counter() - load_started
    _LOG.info("model load wall time: %.2fs (includes HF cache hit or download)", load_seconds)

    # Make inference and log time taken
    forward_started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(batch)
    forward_seconds = time.perf_counter() - forward_started
    _LOG.info("forward wall time: %.2fs", forward_seconds)

    _assert_output_time_dim_is_one(prediction)

    output_surf_shape = tuple(prediction.surf_vars["2t"].shape)
    output_atmos_shape = tuple(prediction.atmos_vars["t"].shape)
    peak_rss_bytes = _peak_rss_bytes()
    _LOG.info(
        "output shapes: surf 2t=%s atmos t=%s (T==1 as expected)",
        output_surf_shape,
        output_atmos_shape,
    )
    _LOG.info("peak RSS: %.2f MiB (%d bytes)", peak_rss_bytes / (1024 * 1024), peak_rss_bytes)
    _LOG.warning(_NO_SKILL_BANNER_BOTTOM)

    return ToyForwardResult(
        input_surf_shape=input_surf_shape,
        input_atmos_shape=input_atmos_shape,
        output_surf_shape=output_surf_shape,
        output_atmos_shape=output_atmos_shape,
        load_seconds=load_seconds,
        forward_seconds=forward_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    run_toy_forward()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
