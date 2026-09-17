"""WP5b: persist fp32 baseline forecasts + kernel profile + measurement floor.

CPU dry-run (``--cpu-dry-run``): ``AuroraSmallPretrained`` on ``SyntheticSource``,
1 init × 2 steps. Exercises the unattended sequence — env JSON, 3-forward
warm-up, one-step ``torch.profiler`` trace, baseline-vs-baseline floor,
per-lead zarr persist, reload round-trip, Q1-shaped RMSE tables vs itself
(RMSE≈0). Not skill.

GPU campaign: same sequence on ``aurora-finetuned`` with the n=30 spread TOML,
``offload_to_cpu=True``, then re-score persisted leads against HRES-T0.
Writes Q1-shaped ``rmse_by_init.csv`` / ``rmse_by_lead.csv``: dry-run vs
itself (RMSE≈0); GPU vs HRES-T0. Compare those tables to 5a in
post-processing — this script does not assert against a reference CSV.
Do not book the box until the CPU dry-run is green.

Example::

    uv sync --extra forecast
    uv run --extra forecast python scripts/save_baseline_forecasts.py \\
        --cpu-dry-run --tag wiring-baselines

    uv run --extra forecast python scripts/save_baseline_forecasts.py \\
        --rollout-config configs/hres_t0_2022_spread_rollout.toml \\
        --output-dir /path/to/nfs/aurora-baselines \\
        --tag wp5b-baselines-n30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from aurora import Aurora, Batch

from aurora_inference.contract import ModelSpec, validate_batch
from aurora_inference.data.hres_t0 import (
    HresT0Source,
    _read_zarr_time_index,
    open_local_zarr,
)
from aurora_inference.data.static_vars import get_hres_t0_static
from aurora_inference.data.synthetic import SyntheticSource
from aurora_inference.evaluation.baselines import (
    FIELDS_FULL,
    FIELDS_HEADLINE,
    FLOOR_LEAD_HOURS,
    HEADLINE_SLICES,
    PersistFields,
    headline_aurora_dataset,
    init_forecast_dir,
    lead_zarr_path,
    read_lead_forecast,
    select_persist_fields,
    write_lead_forecast,
)
from aurora_inference.evaluation.evaluation_schedule import (
    EvalSchedule,
    InitPair,
    SpliceCoverageError,
    assert_times_available,
    build_eval_schedule,
    campaign_n_inits,
    load_eval_schedule_config,
)
from aurora_inference.evaluation.fidelity import compute_rmse_score_variant_vs_baseline
from aurora_inference.evaluation.grids import (
    align_truth_to_forecast,
    analysis_dataset,
    as_naive_datetime,
    batch_to_dataset,
    rmse_rows_from_mse,
)
from aurora_inference.evaluation.metrics import MSE
from aurora_inference.evaluation.tables import load_rmse_rows, write_rmse_tables
from aurora_inference.evaluation.variants import VARIANTS, VariantConfig
from aurora_inference.inference.forward import run_rollout
from aurora_inference.logging import configure_run_logging, normalize_run_tag, run_artifact_dir
from aurora_inference.model.loader import load_model
from aurora_inference.runlog import (
    MemoryRecord,
    RunIdentity,
    RunLog,
    TimingRecord,
    checksums_for_pred,
    collect_env,
    measure_cuda_elapsed_ms,
    reset_vram_peak,
    snapshot_memory,
    write_run_log,
)

_LOG = logging.getLogger(__name__)

_OUTPUT_DIR = Path("outputs")
_DEFAULT_GPU_ROLLOUT = Path("configs/hres_t0_2022_spread_rollout.toml")
_DRY_RUN_DEVICE = "cpu"
_DRY_RUN_MODEL = "aurora-small-pretrained"
_DRY_RUN_STEPS = 2
_DRY_RUN_INIT = datetime(2022, 1, 1, 12, 0)
_DRY_RUN_HEIGHT = 32
_DRY_RUN_WIDTH = 64
_DRY_RUN_SEED = 42
_GPU_DEVICE = "cuda"
_GPU_MODEL = "aurora-finetuned"
_WARMUP_FORWARDS = 3
_RMSE_NEAR_ZERO = 1e-5
_MIB = 1024 * 1024

_NO_SKILL_BANNER = (
    "=== NO FORECAST SKILL === CPU dry-run uses AuroraSmallPretrained + "
    "SyntheticSource. RMSE≈0 only proves WP5b wiring, not forecast quality."
)
_TEARDOWN_REMINDER = (
    "WP5b persist finished. scp artifacts, terminate the GPU instance, and "
    "confirm billing stopped. An attached NFS volume still bills after terminate."
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-dry-run",
        action="store_true",
        help="WP5b wiring check: small model on CPU (no splice, no GPU)",
    )
    parser.add_argument(
        "--rollout-config",
        type=Path,
        default=None,
        help=f"GPU campaign TOML (default: {_DEFAULT_GPU_ROLLOUT})",
    )
    parser.add_argument(
        "--fields",
        choices=(FIELDS_FULL, FIELDS_HEADLINE),
        default=FIELDS_FULL,
        help="what to persist: full weather fields, or the 8 Q1 headline slices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="artefact root (default: outputs/). Use an NFS mount to keep zarrs off instance disk.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes logs and zarrs under <output-dir>/<tag>/",
    )
    parser.add_argument(
        "--max-inits",
        type=int,
        default=None,
        help="cap unique inits after the TOML schedule (GPU smoke; default: all)",
    )
    return parser.parse_args(argv)


def _lookup_variant(name: str) -> VariantConfig | None:
    return VARIANTS.get(name)


def _apply_hygiene(*, device: str) -> None:
    """Set the recorded baseline flags. CPU has no cuDNN autotune knob that matters."""
    torch.set_float32_matmul_precision("highest")
    if device == _GPU_DEVICE and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _sku(device: str) -> str:
    if device != _GPU_DEVICE or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _scalar(value: object) -> float:
    return float(np.asarray(value).item())


def _jsonable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        level = row["level"]
        if level is not None and not (isinstance(level, float) and np.isnan(level)):
            level_out: int | None = int(level)
        else:
            level_out = None
        payload.append(
            {
                "init_id": int(row["init_id"]),
                "init_time": str(row["init_time"]),
                "lead_hours": int(row["lead_hours"]),
                "variable": str(row["variable"]),
                "level": level_out,
                "rmse": float(row["rmse"]),
            }
        )
    return payload


def _write_session_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rmse_dataset_to_rows(
    rmse: xr.Dataset,
    *,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
) -> list[dict[str, Any]]:
    """Flatten a per-variable RMSE dataset to the Q1 long-form table rows."""
    rows: list[dict[str, Any]] = []
    init_iso = init_time.isoformat()
    for name, data_array in rmse.data_vars.items():
        if "level" in data_array.dims:
            for level in data_array["level"].values:
                rows.append(
                    {
                        "init_id": init_id,
                        "init_time": init_iso,
                        "lead_hours": lead_hours,
                        "variable": name,
                        "level": int(level),
                        "rmse": _scalar(data_array.sel(level=level)),
                    }
                )
        else:
            rows.append(
                {
                    "init_id": init_id,
                    "init_time": init_iso,
                    "lead_hours": lead_hours,
                    "variable": name,
                    "level": None,
                    "rmse": _scalar(data_array),
                }
            )
    return rows


def _max_rmse(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return max(float(row["rmse"]) for row in rows)


def _write_skill_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write ``rmse_by_init.csv`` and ``rmse_by_lead.csv`` (same schema as ``eval_rmse.py``)."""
    write_rmse_tables(rows, output_dir)
    _LOG.info("wrote RMSE tables under %s (%s rows)", output_dir, len(rows))


def _persist_variables(spec: ModelSpec, fields: PersistFields) -> list[str]:
    if fields == FIELDS_HEADLINE:
        return [slice_.key for slice_ in HEADLINE_SLICES]
    return [*spec.surf_vars, *spec.atmos_vars]


def _leads_for_floor(n_steps: int, step_hours: int) -> list[int]:
    """Days 1/5/10 when the rollout is long enough; otherwise every produced lead."""
    requested = [lead for lead in FLOOR_LEAD_HOURS if lead <= n_steps * step_hours]
    if requested:
        return requested
    return [step_index * step_hours for step_index in range(1, n_steps + 1)]


def _profiler_activities(device: str) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == _GPU_DEVICE and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _one_forward(model: Aurora, batch: Batch) -> list[Batch]:
    return run_rollout(model, batch, steps=1, offload_to_cpu=True)


def _warmup(model: Aurora, batch: Batch) -> None:
    for index in range(_WARMUP_FORWARDS):
        preds = _one_forward(model, batch)
        _LOG.info("warm-up forward %s/%s discarded", index + 1, _WARMUP_FORWARDS)
        del preds


def _profile_one_forward(model: Aurora, batch: Batch, *, device: str, trace_path: Path) -> None:
    """One step only — do not profile the rollout. Trace is gitignored under outputs/."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=_profiler_activities(device),
        record_shapes=True,
        acc_events=True,
    ) as prof:
        preds = _one_forward(model, batch)
    del preds
    prof.export_chrome_trace(str(trace_path))
    _LOG.info("wrote profiler trace %s", trace_path)


def _timed_rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    *,
    device: str,
) -> tuple[list[Batch], float, float | None]:
    started = time.perf_counter()
    preds, cuda_ms = measure_cuda_elapsed_ms(
        lambda: run_rollout(model, batch, steps, offload_to_cpu=True),
        device=device,
    )
    return preds, time.perf_counter() - started, cuda_ms


def _log_after_persist_rollout(memory: MemoryRecord, *, init_id: int) -> None:
    """One seam: after persist rollout. Peak is since the reset, not process start."""
    rss_mib = memory.peak_rss_bytes / _MIB
    if memory.max_memory_allocated is None:
        _LOG.info(
            "init_id=%s after persist rollout: peak RSS=%.2f MiB (VRAM n/a)",
            init_id,
            rss_mib,
        )
        return
    reserved = 0 if memory.max_memory_reserved is None else memory.max_memory_reserved
    _LOG.info(
        "init_id=%s after persist rollout: peak RSS=%.2f MiB  "
        "VRAM allocated peak=%.2f MiB  VRAM reserved peak=%.2f MiB",
        init_id,
        rss_mib,
        memory.max_memory_allocated / _MIB,
        reserved / _MIB,
    )


def _persist_rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    *,
    device: str,
    init_id: int,
) -> tuple[list[Batch], float, float | None, MemoryRecord]:
    """Reset CUDA peak, roll out, snapshot once (OOM check for 40-step persist)."""
    reset_vram_peak(device)
    preds, wall_s, cuda_ms = _timed_rollout(model, batch, steps, device=device)
    memory = snapshot_memory(device)
    _log_after_persist_rollout(memory, init_id=init_id)
    return preds, wall_s, cuda_ms, memory


def _measurement_floor(
    model: Aurora,
    batch: Batch,
    *,
    spec: ModelSpec,
    fields: PersistFields,
    n_steps: int,
    init_id: int,
    init_time: datetime,
    device: str,
) -> dict[str, Any]:
    """Two identical fp32 rollouts; RMSE at day 1/5/10 (or every dry-run lead)."""
    variables = _persist_variables(spec, fields)
    step_hours = spec.input_timestep_hours
    leads = _leads_for_floor(n_steps, step_hours)
    first, first_s, _ = _timed_rollout(model, batch, n_steps, device=device)
    second, second_s, _ = _timed_rollout(model, batch, n_steps, device=device)
    rows: list[dict[str, Any]] = []
    for lead_hours in leads:
        step_index = lead_hours // step_hours
        baseline = select_persist_fields(batch_to_dataset(first[step_index - 1]), fields)
        variant = select_persist_fields(batch_to_dataset(second[step_index - 1]), fields)
        rmse = compute_rmse_score_variant_vs_baseline(
            baseline=baseline,
            variant=variant,
            variables=variables,
        )
        rows.extend(
            _rmse_dataset_to_rows(
                rmse,
                init_id=init_id,
                init_time=init_time,
                lead_hours=lead_hours,
            )
        )
    del first, second
    max_rmse = _max_rmse(rows)
    warnings: list[str] = []
    if max_rmse > _RMSE_NEAR_ZERO:
        warnings.append(
            "non-zero measurement floor; cudnn.benchmark autotuning is the usual GPU suspect"
        )
    _LOG.info(
        "measurement floor: n_steps=%s leads=%s max_rmse=%.4g (rollout wall %.2fs + %.2fs)",
        n_steps,
        leads,
        max_rmse,
        first_s,
        second_s,
    )
    return {
        "init_id": init_id,
        "init_time": init_time.isoformat(),
        "n_steps": n_steps,
        "lead_hours_requested": list(FLOOR_LEAD_HOURS),
        "lead_hours_recorded": leads,
        "max_rmse": max_rmse,
        "rows": _jsonable_rows(rows),
        "warnings": warnings,
    }


def _persist_preds(
    preds: list[Batch],
    *,
    init_dir: Path,
    spec: ModelSpec,
    fields: PersistFields,
) -> None:
    step_hours = spec.input_timestep_hours
    for step_index, pred in enumerate(preds, start=1):
        lead_hours = step_index * step_hours
        dataset = select_persist_fields(batch_to_dataset(pred), fields)
        write_lead_forecast(dataset, lead_zarr_path(init_dir, lead_hours))
    _LOG.info("persisted %s leads under %s (fields=%s)", len(preds), init_dir, fields)


def _round_trip_vs_preds(
    preds: list[Batch],
    *,
    init_dir: Path,
    spec: ModelSpec,
    fields: PersistFields,
    init_id: int,
    init_time: datetime,
) -> list[dict[str, Any]]:
    """Reload each lead and RMSE against the in-memory forecast (CPU wiring)."""
    variables = _persist_variables(spec, fields)
    step_hours = spec.input_timestep_hours
    rows: list[dict[str, Any]] = []
    for step_index, pred in enumerate(preds, start=1):
        lead_hours = step_index * step_hours
        original = select_persist_fields(batch_to_dataset(pred), fields)
        loaded = read_lead_forecast(lead_zarr_path(init_dir, lead_hours))
        rmse = compute_rmse_score_variant_vs_baseline(
            baseline=original,
            variant=loaded,
            variables=variables,
        )
        rows.extend(
            _rmse_dataset_to_rows(
                rmse,
                init_id=init_id,
                init_time=init_time,
                lead_hours=lead_hours,
            )
        )
    return rows


def _score_forecast_vs_truth(
    forecast: xr.Dataset,
    *,
    source: HresT0Source,
    valid_time: datetime,
    init_id: int,
    init_time: datetime,
    lead_hours: int,
    spec: ModelSpec,
    fields: PersistFields,
) -> list[dict[str, Any]]:
    analysis = analysis_dataset(source, valid_time, spec)
    if fields == FIELDS_HEADLINE:
        rows: list[dict[str, Any]] = []
        for slice_ in HEADLINE_SLICES:
            aurora = headline_aurora_dataset(forecast, slice_)
            truth_field = analysis[slice_.aurora_name]
            if slice_.level is not None:
                truth_field = truth_field.sel(level=[slice_.level])
            truth = align_truth_to_forecast(aurora, truth_field.to_dataset(name=slice_.aurora_name))
            mse = MSE().compute_chunk(aurora, truth)
            rows.extend(
                rmse_rows_from_mse(
                    mse,
                    init_id=init_id,
                    init_time=init_time,
                    lead_hours=lead_hours,
                )
            )
        return rows
    truth = align_truth_to_forecast(forecast, analysis)
    shared = [name for name in forecast.data_vars if name in truth.data_vars]
    mse = MSE().compute_chunk(forecast[shared], truth[shared])
    return rmse_rows_from_mse(
        mse,
        init_id=init_id,
        init_time=init_time,
        lead_hours=lead_hours,
    )


def _load_campaign(rollout_config_path: Path) -> EvalSchedule:
    config = load_eval_schedule_config(rollout_config_path)
    schedule = build_eval_schedule(config)
    group = open_local_zarr(config.splice_path)
    assert_times_available(schedule.needed_times, _read_zarr_time_index(group))
    _LOG.info(
        "campaign: %s inits × %s steps (%s unique times) ⊆ %s",
        campaign_n_inits(config),
        config.n_rollout_steps,
        len(schedule.needed_times),
        config.splice_path,
    )
    return schedule


def _open_hres_source(splice_path: Path) -> HresT0Source:
    return HresT0Source(ZARR_DATA=open_local_zarr(splice_path), STATIC_VARS=get_hres_t0_static())


def _write_init_run_log(
    output_dir: Path,
    *,
    env: Any,
    variant_name: str,
    init_id: int,
    init_time: datetime,
    device: str,
    n_steps: int,
    model_name: str,
    cpu_dry_run: bool,
    load_s: float,
    rollout_s: float,
    rollout_ms: float | None,
    preds: list[Batch],
    memory: MemoryRecord,
    max_rmse: float | None,
    seed: int | None,
) -> None:
    run_path = output_dir / f"run_init-{init_id}.json"
    write_run_log(
        run_path,
        RunLog(
            env=env,
            run=RunIdentity(
                variant=variant_name,
                init_id=init_id,
                init_time=init_time.isoformat(),
                sku=_sku(device),
                device=device,
                batch_size=1,
                n_steps=n_steps,
                seed=seed,
                model_name=model_name,
                offload_to_cpu=True,
                cpu_dry_run=cpu_dry_run,
            ),
            timing=TimingRecord(
                model_load_s=load_s,
                full_rollout_wall_s=rollout_s,
                full_rollout_ms=rollout_ms,
                per_step_ms=[None] * n_steps,
            ),
            memory=memory,
            checksums=[
                checksums_for_pred(pred, step=step_index)
                for step_index, pred in enumerate(preds, start=1)
            ],
            max_rmse_variant_vs_baseline=max_rmse,
        ),
    )
    _LOG.info("wrote run JSON %s", run_path)


def _session_payload(
    *,
    env: Any,
    fields: PersistFields,
    device: str,
    cpu_dry_run: bool,
    model_name: str,
    n_steps: int,
    floor: dict[str, Any],
    trace_name: str,
) -> dict[str, Any]:
    return {
        "cpu_dry_run": cpu_dry_run,
        "device": device,
        "env": env.model_dump(mode="json"),
        "fields": fields,
        "floor": floor,
        "model_name": model_name,
        "n_steps": n_steps,
        "profile_trace": trace_name,
        "sku": _sku(device),
        "variant": "fp32-baseline",
        "warmup_forwards": _WARMUP_FORWARDS,
    }


def _run_cpu_dry_run(*, variant: VariantConfig, output_dir: Path, fields: PersistFields) -> int:
    _LOG.warning(_NO_SKILL_BANNER)
    _apply_hygiene(device=_DRY_RUN_DEVICE)
    spec = variant.spec
    source = SyntheticSource(
        height=_DRY_RUN_HEIGHT,
        width=_DRY_RUN_WIDTH,
        seed=_DRY_RUN_SEED,
    )
    batch = source.load(_DRY_RUN_INIT, spec)
    validate_batch(batch, spec)
    batch = batch.to(_DRY_RUN_DEVICE)

    load_started = time.perf_counter()
    model: Aurora = load_model(_DRY_RUN_MODEL, device=_DRY_RUN_DEVICE)
    load_s = time.perf_counter() - load_started
    _LOG.info("loaded %s on %s in %.2fs", _DRY_RUN_MODEL, _DRY_RUN_DEVICE, load_s)

    env = collect_env()
    session_path = output_dir / "session.json"
    _write_session_json(
        session_path,
        _session_payload(
            env=env,
            fields=fields,
            device=_DRY_RUN_DEVICE,
            cpu_dry_run=True,
            model_name=_DRY_RUN_MODEL,
            n_steps=_DRY_RUN_STEPS,
            floor={},
            trace_name="trace.json",
        ),
    )
    _LOG.info("wrote session env %s", session_path)

    _warmup(model, batch)
    _profile_one_forward(
        model,
        batch,
        device=_DRY_RUN_DEVICE,
        trace_path=output_dir / "trace.json",
    )
    floor = _measurement_floor(
        model,
        batch,
        spec=spec,
        fields=fields,
        n_steps=_DRY_RUN_STEPS,
        init_id=0,
        init_time=_DRY_RUN_INIT,
        device=_DRY_RUN_DEVICE,
    )
    _write_session_json(
        session_path,
        _session_payload(
            env=env,
            fields=fields,
            device=_DRY_RUN_DEVICE,
            cpu_dry_run=True,
            model_name=_DRY_RUN_MODEL,
            n_steps=_DRY_RUN_STEPS,
            floor=floor,
            trace_name="trace.json",
        ),
    )

    preds, persist_s, persist_ms, persist_memory = _persist_rollout(
        model,
        batch,
        _DRY_RUN_STEPS,
        device=_DRY_RUN_DEVICE,
        init_id=0,
    )
    init_dir = init_forecast_dir(output_dir, 0)
    _persist_preds(preds, init_dir=init_dir, spec=spec, fields=fields)
    round_trip_rows = _round_trip_vs_preds(
        preds,
        init_dir=init_dir,
        spec=spec,
        fields=fields,
        init_id=0,
        init_time=_DRY_RUN_INIT,
    )
    max_round_trip = _max_rmse(round_trip_rows)
    _LOG.info("persist round-trip max_rmse=%.4g (vs itself; not skill)", max_round_trip)
    _write_skill_tables(round_trip_rows, output_dir)

    _write_init_run_log(
        output_dir,
        env=env,
        variant_name=variant.name,
        init_id=0,
        init_time=_DRY_RUN_INIT,
        device=_DRY_RUN_DEVICE,
        n_steps=_DRY_RUN_STEPS,
        model_name=_DRY_RUN_MODEL,
        cpu_dry_run=True,
        load_s=load_s,
        rollout_s=persist_s,
        rollout_ms=persist_ms,
        preds=preds,
        memory=persist_memory,
        max_rmse=max_round_trip,
        seed=_DRY_RUN_SEED,
    )
    _write_session_json(
        session_path,
        _session_payload(
            env=env,
            fields=fields,
            device=_DRY_RUN_DEVICE,
            cpu_dry_run=True,
            model_name=_DRY_RUN_MODEL,
            n_steps=_DRY_RUN_STEPS,
            floor=floor,
            trace_name="trace.json",
        ),
    )
    _LOG.warning(_NO_SKILL_BANNER)

    if not np.isfinite(max_round_trip) or max_round_trip > _RMSE_NEAR_ZERO:
        _LOG.error(
            "dry-run persist round-trip max=%.4g; expected ≈ 0 (atol=%g)",
            max_round_trip,
            _RMSE_NEAR_ZERO,
        )
        return 1
    return 0


def _run_gpu_campaign(
    *,
    variant: VariantConfig,
    output_dir: Path,
    fields: PersistFields,
    rollout_config: Path,
    max_inits: int | None,
) -> int:
    _apply_hygiene(device=_GPU_DEVICE)
    spec = variant.spec
    try:
        schedule = _load_campaign(rollout_config)
    except (FileNotFoundError, SpliceCoverageError, TypeError) as exc:
        _LOG.error("%s", exc)
        return 1

    source = _open_hres_source(schedule.config.splice_path)
    load_started = time.perf_counter()
    model = load_model(_GPU_MODEL, device=_GPU_DEVICE)
    load_s = time.perf_counter() - load_started
    _LOG.info("loaded %s on %s in %.2fs", _GPU_MODEL, _GPU_DEVICE, load_s)
    env = collect_env()

    init_pairs = schedule.init_pairs
    if max_inits is not None:
        init_pairs = init_pairs[:max_inits]
        _LOG.info("capping campaign at %s inits (of %s)", len(init_pairs), len(schedule.init_pairs))
    if not init_pairs:
        _LOG.error("campaign has no inits")
        return 1
    first = init_pairs[0]
    first_init = as_naive_datetime(first.init_time)
    batch = source.load(first_init, spec)
    validate_batch(batch, spec)
    batch = batch.to(_GPU_DEVICE)

    _warmup(model, batch)
    _profile_one_forward(
        model,
        batch,
        device=_GPU_DEVICE,
        trace_path=output_dir / "trace.json",
    )
    floor = _measurement_floor(
        model,
        batch,
        spec=spec,
        fields=fields,
        n_steps=first.n_rollout_steps,
        init_id=first.init_id,
        init_time=first_init,
        device=_GPU_DEVICE,
    )
    session_path = output_dir / "session.json"
    _write_session_json(
        session_path,
        _session_payload(
            env=env,
            fields=fields,
            device=_GPU_DEVICE,
            cpu_dry_run=False,
            model_name=_GPU_MODEL,
            n_steps=first.n_rollout_steps,
            floor=floor,
            trace_name="trace.json",
        ),
    )
    del batch

    rows = load_rmse_rows(output_dir)
    done_ids = {int(row["init_id"]) for row in rows}
    for init_pair in init_pairs:
        if init_pair.init_id in done_ids:
            _LOG.info("skip init_id=%s (already scored)", init_pair.init_id)
            continue
        scored = _persist_gpu_init(
            init_pair,
            source=source,
            model=model,
            spec=spec,
            fields=fields,
            output_dir=output_dir,
            env=env,
            variant_name=variant.name,
            load_s=load_s,
        )
        rows.extend(scored)
        _write_skill_tables(rows, output_dir)
        done_ids.add(init_pair.init_id)

    if not rows:
        _LOG.error("no RMSE rows produced")
        return 1
    _write_skill_tables(rows, output_dir)
    _LOG.info(_TEARDOWN_REMINDER)
    return 0


def _persist_gpu_init(
    init_pair: InitPair,
    *,
    source: HresT0Source,
    model: Aurora,
    spec: ModelSpec,
    fields: PersistFields,
    output_dir: Path,
    env: Any,
    variant_name: str,
    load_s: float,
) -> list[dict[str, Any]]:
    init_time = as_naive_datetime(init_pair.init_time)
    batch = source.load(init_time, spec)
    validate_batch(batch, spec)
    batch = batch.to(_GPU_DEVICE)
    _LOG.info(
        "init_id=%s init_time=%s steps=%s",
        init_pair.init_id,
        init_time.isoformat(),
        init_pair.n_rollout_steps,
    )
    preds, persist_s, persist_ms, persist_memory = _persist_rollout(
        model,
        batch,
        init_pair.n_rollout_steps,
        device=_GPU_DEVICE,
        init_id=init_pair.init_id,
    )
    del batch
    init_dir = init_forecast_dir(output_dir, init_pair.init_id)
    _persist_preds(preds, init_dir=init_dir, spec=spec, fields=fields)

    scored: list[dict[str, Any]] = []
    step_hours = spec.input_timestep_hours
    for step_index in range(1, init_pair.n_rollout_steps + 1):
        lead_hours = step_index * step_hours
        valid_time = init_time + timedelta(hours=lead_hours)
        forecast = read_lead_forecast(lead_zarr_path(init_dir, lead_hours))
        scored.extend(
            _score_forecast_vs_truth(
                forecast,
                source=source,
                valid_time=valid_time,
                init_id=init_pair.init_id,
                init_time=init_time,
                lead_hours=lead_hours,
                spec=spec,
                fields=fields,
            )
        )
    _LOG.info("init_id=%s scored vs HRES-T0 (%s rows)", init_pair.init_id, len(scored))

    _write_init_run_log(
        output_dir,
        env=env,
        variant_name=variant_name,
        init_id=init_pair.init_id,
        init_time=init_time,
        device=_GPU_DEVICE,
        n_steps=init_pair.n_rollout_steps,
        model_name=_GPU_MODEL,
        cpu_dry_run=False,
        load_s=load_s,
        rollout_s=persist_s,
        rollout_ms=persist_ms,
        preds=preds,
        memory=persist_memory,
        max_rmse=None,
        seed=None,
    )
    del preds
    return scored


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

    variant = _lookup_variant("fp32-baseline")
    if variant is None:
        print("VARIANTS is missing fp32-baseline", file=sys.stderr)
        return 2

    fields: PersistFields = args.fields
    output_dir = run_artifact_dir(args.output_dir.expanduser(), tag)
    configure_run_logging(output_dir / "save_baseline_forecasts.log", tag=tag)

    if args.cpu_dry_run:
        if args.rollout_config is not None:
            print(
                "--cpu-dry-run does not take --rollout-config (uses SyntheticSource)",
                file=sys.stderr,
            )
            return 2
        _LOG.info(
            "WP5b baseline persist CPU dry-run; log file: %s",
            output_dir / "save_baseline_forecasts.log",
        )
        return _run_cpu_dry_run(variant=variant, output_dir=output_dir, fields=fields)

    rollout_config = _DEFAULT_GPU_ROLLOUT if args.rollout_config is None else args.rollout_config
    if not torch.cuda.is_available():
        print(
            "CUDA is required for the GPU campaign. Pass --cpu-dry-run for the wiring check.",
            file=sys.stderr,
        )
        return 2

    _LOG.info(
        "WP5b baseline persist GPU campaign; log file: %s",
        output_dir / "save_baseline_forecasts.log",
    )
    return _run_gpu_campaign(
        variant=variant,
        output_dir=output_dir,
        fields=fields,
        rollout_config=rollout_config,
        max_inits=args.max_inits,
    )


if __name__ == "__main__":
    raise SystemExit(main())
