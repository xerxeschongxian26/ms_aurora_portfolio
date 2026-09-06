"""Many HRES-T0 inits × rollout → mean RMSE vs lead time.

Skill path: unique 00/12 UTC inits from rollout.toml, B=1, score on the instance,
egress the RMSE table and logs. Does not write forecast cubes or PNGs.

``real_forecast.py`` remains the GPU/VRAM/PNG / ``--batch-size`` probe.

Example (toy: 2 inits × 3 steps):

    uv run --extra forecast python scripts/toy_hres_t0_splice.py --skip-download
    uv run --extra forecast python scripts/eval_rmse.py \\
        --rollout-config configs/hres_t0_toy_rollout.toml --tag toy-rmse
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from aurora import Aurora, Batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec, validate_batch
from aurora_inference.data.hres_t0 import (
    HresT0Source,
    _read_zarr_time_index,
    open_local_zarr,
)
from aurora_inference.data.static_vars import get_hres_t0_static
from aurora_inference.evaluation.evaluation_schedule import (
    EvalSchedule,
    InitPair,
    SpliceCoverageError,
    assert_times_available,
    build_eval_schedule,
    load_eval_schedule_config,
)
from aurora_inference.evaluation.grids import as_naive_datetime, score_lead_rows
from aurora_inference.evaluation.tables import load_rmse_rows, write_rmse_tables
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
_DEFAULT_ROLLOUT = Path("configs/hres_t0_toy_rollout.toml")
_OUTPUT_DIR = Path("outputs")
_MIB = 1024 * 1024


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
        "--rollout-config",
        type=Path,
        default=_DEFAULT_ROLLOUT,
        help="TOML campaign (n_inits, n_rollout_steps, splice_path)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes logs and the RMSE table under outputs/<tag>/",
    )
    parser.add_argument(
        "--max-inits",
        type=int,
        default=None,
        help="cap unique inits after the TOML schedule (toy smoke; default: all)",
    )
    return parser.parse_args(argv)


def _load_campaign(rollout_config_path: Path) -> EvalSchedule:
    """Load rollout.toml, build ``EvalSchedule``, assert the local splice covers it."""
    config = load_eval_schedule_config(rollout_config_path)
    schedule = build_eval_schedule(config)
    group = open_local_zarr(config.splice_path)
    assert_times_available(schedule.needed_times, _read_zarr_time_index(group))
    _LOG.info(
        "campaign: %s inits × %s steps (%s unique times) ⊆ %s",
        config.n_inits,
        config.n_rollout_steps,
        len(schedule.needed_times),
        config.splice_path,
    )
    return schedule


def _open_hres_source(splice_path: Path) -> HresT0Source:
    """Open the local HRES-T0 splice once (shared static vars + zarr group)."""
    return HresT0Source(ZARR_DATA=open_local_zarr(splice_path), STATIC_VARS=get_hres_t0_static())


def _load_model_once() -> Aurora:
    """Load ``aurora-finetuned`` once; reuse across all inits."""
    model = load_model(model_name=_MODEL_NAME, device=_DEVICE)
    _LOG.info("loaded %s on %s", _MODEL_NAME, _DEVICE)
    return model


def _rollout_offload(model: Aurora, batch: Batch, steps: int) -> list[Batch]:
    """Roll out ``steps`` leads; copy each pred to CPU without mutating GPU tensors."""
    return run_rollout(model, batch, steps, offload_to_cpu=True)


def _score_lead(
    pred: Batch,
    valid_time: datetime,
    source: HresT0Source,
    *,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
    spec: ModelSpec = AURORA_PRETRAINED_SPEC,
) -> list[dict[str, Any]]:
    """Load one truth state at ``valid_time``, RMSE vs ``pred``, drop the grids."""
    return score_lead_rows(
        pred,
        source=source,
        valid_time=as_naive_datetime(valid_time),
        init_id=init_id,
        init_time=as_naive_datetime(init_time),
        lead_hours=lead_hours,
        spec=spec,
    )


def _run_init(
    init_pair: InitPair,
    *,
    source: HresT0Source,
    model: Aurora,
    spec: ModelSpec = AURORA_PRETRAINED_SPEC,
) -> list[dict[str, Any]]:
    """One unique init: load T=2 input, rollout, score each lead, drop tensors."""
    init_time = as_naive_datetime(init_pair.init_time)
    batch = source.load(init_time, spec)
    validate_batch(batch, spec)
    _LOG.info(
        "init_id=%s init_time=%s steps=%s",
        init_pair.init_id,
        init_time.isoformat(),
        init_pair.n_rollout_steps,
    )
    _reset_vram_peak(_DEVICE)
    batch = batch.to(_DEVICE)
    _log_mem(f"init_id={init_pair.init_id} after batch.to")

    _reset_vram_peak(_DEVICE)
    rollout_started = time.perf_counter()
    predictions = _rollout_offload(model, batch, init_pair.n_rollout_steps)
    _LOG.info(
        "init_id=%s run_rollout wall time: %.2fs for %d steps",
        init_pair.init_id,
        time.perf_counter() - rollout_started,
        init_pair.n_rollout_steps,
    )
    _log_mem(f"init_id={init_pair.init_id} after rollout")
    del batch

    rows: list[dict[str, Any]] = []
    step_hours = spec.input_timestep_hours
    for step_index, pred in enumerate(predictions, start=1):
        lead_hours = step_index * step_hours
        valid_time = init_time + timedelta(hours=lead_hours)
        lead_rows = _score_lead(
            pred,
            valid_time,
            source,
            init_id=init_pair.init_id,
            init_time=init_time,
            lead_hours=lead_hours,
            spec=spec,
        )
        rows.extend(lead_rows)
        _LOG.info(
            "scored init_id=%s lead=%sh n_rows=%s",
            init_pair.init_id,
            lead_hours,
            len(lead_rows),
        )
    del predictions
    _log_mem(f"init_id={init_pair.init_id} after score")
    return rows


def _write_mean_rmse_table(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """Average RMSE over inits, grouped by lead time and variable."""
    out_path = write_rmse_tables(rows, output_dir)
    _LOG.info("wrote RMSE tables under %s (%s rows)", output_dir, len(rows))
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tag = normalize_run_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.max_inits is not None and args.max_inits < 1:
        print(f"max-inits must be >= 1 (got {args.max_inits})", file=sys.stderr)
        return 2

    output_dir = run_artifact_dir(_OUTPUT_DIR, tag)
    configure_run_logging(output_dir / "eval_rmse.log", tag=tag)
    _LOG.info("Evaluate RMSE; log file: %s", output_dir / "eval_rmse.log")

    try:
        schedule = _load_campaign(args.rollout_config)
    except (FileNotFoundError, SpliceCoverageError, TypeError) as exc:
        _LOG.error("%s", exc)
        return 1

    if not torch.cuda.is_available():
        _LOG.error("device=%s but CUDA is unavailable", _DEVICE)
        return 1

    _reset_vram_peak(_DEVICE)
    source = _open_hres_source(schedule.config.splice_path)
    _log_mem("after source open")

    _reset_vram_peak(_DEVICE)
    load_started = time.perf_counter()
    model = _load_model_once()
    _LOG.info("model load wall time: %.2fs", time.perf_counter() - load_started)
    _log_mem("after model load")

    init_pairs = schedule.init_pairs
    if args.max_inits is not None:
        init_pairs = init_pairs[: args.max_inits]
        _LOG.info("capping campaign at %s inits (of %s)", len(init_pairs), schedule.config.n_inits)

    rows = load_rmse_rows(output_dir)
    done_ids = {int(row["init_id"]) for row in rows}
    for init_pair in init_pairs:
        if init_pair.init_id in done_ids:
            _LOG.info("skip init_id=%s (already scored)", init_pair.init_id)
            continue
        rows.extend(_run_init(init_pair, source=source, model=model))
        _write_mean_rmse_table(rows, output_dir)
        done_ids.add(init_pair.init_id)

    if not rows:
        _LOG.error("no RMSE rows produced")
        return 1
    _write_mean_rmse_table(rows, output_dir)
    _log_mem("after campaign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
