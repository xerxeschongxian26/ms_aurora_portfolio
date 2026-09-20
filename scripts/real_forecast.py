"""Real forecast on a GPU: HRES-T0 → Aurora() rollout → one 2t PNG per step.

Aurora() possesses forecast skill. Recall that Aurora(), is AuroraPretrained that
has been fine-tuned on the HRES-T0 dataset.

uv sync --extra forecast
python scripts/real_forecast.py --steps 4 --tag b1-s4
python scripts/real_forecast.py --steps 1 --batch-size 2 --tag b2-s1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
from aurora import Batch
from plot_forecast import plot_batch_2t

from aurora_inference.config import GCS_STORE_LINK
from aurora_inference.contract import AURORA_PRETRAINED_SPEC, validate_batch
from aurora_inference.data.hres_t0 import HresT0Source, open_connection_to_gcs
from aurora_inference.data.static_vars import get_hres_t0_static
from aurora_inference.inference.forward import run_rollout
from aurora_inference.logging import (
    configure_run_logging,
    normalize_run_tag,
    peak_rss_bytes,
    run_artifact_dir,
)
from aurora_inference.model.loader import load_model

_LOG = logging.getLogger(__name__)

_MODEL_NAME = "aurora-finetuned"
_DEVICE = "cuda"
_DEFAULT_STEPS = 4
_INIT_TIME = datetime(2022, 6, 15, 12, 0)
_OUTPUT_DIR = Path("outputs")

_SKILL_BANNER = "=== FORECAST SKILL === Aurora performs a real forecast."
_MIB = 1024 * 1024


def _repeat_batch(batch: Batch, times: int) -> Batch:
    """Stack ``batch`` ``times`` times along dim 0 with real copies, not expand.

    Copies ``surf_vars`` and ``atmos_vars``. Leaves ``static_vars``, ``lat``, and
    ``lon`` shared. Repeats ``metadata.time`` so ``len(time) == B``.
    """
    if times < 1:
        msg = f"times must be >= 1 (got {times})"
        raise ValueError(msg)
    if times == 1:
        return batch

    def _repeat_b(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat(times, *((1,) * (tensor.ndim - 1)))

    return Batch(
        surf_vars={key: _repeat_b(tensor) for key, tensor in batch.surf_vars.items()},
        atmos_vars={key: _repeat_b(tensor) for key, tensor in batch.atmos_vars.items()},
        static_vars=batch.static_vars,
        metadata=replace(batch.metadata, time=tuple(batch.metadata.time) * times),
    )


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _vram_bytes(device: str) -> tuple[int, int] | None:
    """Return ``(allocated_now, allocated_peak)`` in bytes, or ``None`` if not CUDA."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(torch_device)
    index = _cuda_device_index(torch_device)
    return (
        int(torch.cuda.memory_allocated(index)),
        int(torch.cuda.max_memory_allocated(index)),
    )


def _reset_vram_peak(device: str) -> None:
    """Reset the CUDA allocation high-water mark. No-op if not CUDA."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(torch_device)
    torch.cuda.reset_peak_memory_stats(_cuda_device_index(torch_device))


def _log_mem(label: str) -> None:
    """Log host peak RSS and CUDA allocated/peak at a pipeline seam.

    VRAM peak is since the last ``_reset_vram_peak`` (one phase), not process start.
    """
    rss_mib = peak_rss_bytes() / _MIB
    vram = _vram_bytes(_DEVICE)
    if vram is None:
        _LOG.info("%s: peak RSS=%.2f MiB (VRAM n/a, device=%s)", label, rss_mib, _DEVICE)
        return
    now_bytes, peak_bytes = vram
    _LOG.info(
        "%s: peak RSS=%.2f MiB  VRAM now=%.2f MiB  VRAM peak=%.2f MiB",
        label,
        rss_mib,
        now_bytes / _MIB,
        peak_bytes / _MIB,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_STEPS,
        help=f"rollout steps (default {_DEFAULT_STEPS}; WP4 acceptance uses 4)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes logs and PNGs under outputs/<tag>/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "repeat the loaded B=1 batch this many times with real copies "
            "(memory probe; default 1). Not unique inits."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tag = normalize_run_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print(f"batch-size must be >= 1 (got {args.batch_size})", file=sys.stderr)
        return 2
    output_dir = run_artifact_dir(_OUTPUT_DIR, tag)
    configure_run_logging(output_dir / "real_forecast.log", tag=tag)
    _LOG.info(_SKILL_BANNER)
    _LOG.info("log file: %s", output_dir / "real_forecast.log")

    # SystemExit if cuda is unavailable
    if not torch.cuda.is_available():
        _LOG.error("_DEVICE = %s but CUDA device is unavailable", _DEVICE)
        return 1

    # Create a HresT0 Batch object
    zarr_data = open_connection_to_gcs(gcs_store_link=GCS_STORE_LINK)
    static_vars = get_hres_t0_static()
    source = HresT0Source(ZARR_DATA=zarr_data, STATIC_VARS=static_vars)
    batch = source.load(_INIT_TIME, AURORA_PRETRAINED_SPEC)
    batch = _repeat_batch(batch, args.batch_size)
    validate_batch(batch, AURORA_PRETRAINED_SPEC)
    _reset_vram_peak(_DEVICE)
    batch = batch.to(_DEVICE)
    _log_mem("after batch.to")

    _LOG.info(
        "input: surf 2t=%s atmos t=%s, init=%s batch_size=%d device=%s",
        tuple(batch.surf_vars["2t"].shape),
        tuple(batch.atmos_vars["t"].shape),
        _INIT_TIME.isoformat(),
        args.batch_size,
        _DEVICE,
    )

    _reset_vram_peak(_DEVICE)
    load_started = time.perf_counter()
    model = load_model(model_name=_MODEL_NAME, device=_DEVICE)
    _LOG.info("model load wall time: %.2fs", time.perf_counter() - load_started)
    _log_mem("after model load")

    _reset_vram_peak(_DEVICE)
    rollout_started = time.perf_counter()
    predictions = run_rollout(model, batch, steps=args.steps)
    _LOG.info(
        "run_rollout wall time: %.2fs for %d steps",
        time.perf_counter() - rollout_started,
        args.steps,
    )
    _log_mem("after rollout")

    _reset_vram_peak(_DEVICE)
    for i, pred in enumerate(predictions, start=1):
        output_path = output_dir / f"real_forecast_2t_step{i:02d}.png"
        plot_batch_2t(pred, output_path)

    _log_mem("after plots")
    _LOG.warning(_SKILL_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
