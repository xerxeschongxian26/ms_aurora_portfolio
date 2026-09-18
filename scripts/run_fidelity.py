"""Q2 fidelity campaign: named variant vs persisted baseline forecasts.

WP4 CPU dry-run: scores ``fp32-baseline`` against itself on
``AuroraSmallPretrained`` (synthetic 32×64 batch, 1 init, 2 steps). That checks
the seam: registry lookup → rollout → ``batch_to_dataset`` → RMSE helpers →
tables + ``runlog.py`` JSON. RMSE(variant, baseline) must be ≈ 0. Plumbing,
not skill — do not quote those RMSEs.

GPU campaign: roll a named ``VARIANTS`` entry on a rollout TOML, score each
lead against the WP5b **headline** archive (RMSE vs baseline) and HRES-T0
(RMSE vs truth). Pair ``(init_id, lead_hours)`` using persist ``init_time``,
not the screen TOML's 1..6 re-index.

Example::

    uv sync --extra forecast
    uv run --extra forecast python scripts/run_fidelity.py --cpu-dry-run --tag wiring

    uv run --extra forecast python scripts/run_fidelity.py \\
        --baseline-dir /path/to/wp5b-baselines-n30-headline \\
        --rollout-config configs/hres_t0_2022_fidelity_screen_rollout.toml \\
        --output-dir /path/to/nfs/aurora-fidelity --tag screen-fp32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from aurora import Aurora, Batch

from aurora_inference.contract import ModelSpec, validate_batch
from aurora_inference.data.hres_t0 import HresT0Source, _read_zarr_time_index, open_local_zarr
from aurora_inference.data.static_vars import get_hres_t0_static
from aurora_inference.data.synthetic import SyntheticSource
from aurora_inference.evaluation.baselines import (
    FIELDS_HEADLINE,
    init_forecast_dir,
    lead_zarr_path,
    persist_init_ids_by_time,
    read_lead_forecast,
    score_headline_vs_analysis_rows,
    score_headline_vs_baseline_rows,
    select_persist_fields,
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
from aurora_inference.evaluation.grids import analysis_dataset, as_naive_datetime, batch_to_dataset
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
_DEFAULT_GPU_ROLLOUT = Path("configs/hres_t0_2022_fidelity_screen_rollout.toml")
_DRY_RUN_DEVICE = "cpu"
_DRY_RUN_MODEL = "aurora-small-pretrained"
_DRY_RUN_STEPS = 2
_DRY_RUN_INIT = datetime(2022, 1, 1, 12, 0)
_DRY_RUN_HEIGHT = 32
_DRY_RUN_WIDTH = 64
_DRY_RUN_SEED = 42
_RMSE_NEAR_ZERO = 1e-5
_GPU_DEVICE = "cuda"
_WARMUP_FORWARDS = 3
_MIB = 1024 * 1024
_VS_BASELINE = "vs_baseline"
_VS_TRUTH = "vs_truth"

_NO_SKILL_BANNER = (
    "=== NO FORECAST SKILL === CPU dry-run uses AuroraSmallPretrained + "
    "SyntheticSource. RMSE≈0 only proves the Q2 wiring, not forecast quality."
)
_TEARDOWN_REMINDER = (
    "Q2 GPU fidelity finished. scp artifacts, terminate the GPU instance, and "
    "confirm billing stopped. An attached NFS volume still bills after terminate."
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-dry-run",
        action="store_true",
        help="WP4 wiring check: small model vs itself on CPU (no splice, no GPU)",
    )
    parser.add_argument(
        "--variant",
        default="fp32-baseline",
        help="VARIANTS key (dry-run only supports fp32-baseline)",
    )
    parser.add_argument(
        "--rollout-config",
        type=Path,
        default=None,
        help=f"GPU campaign TOML (default: {_DEFAULT_GPU_ROLLOUT})",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="WP5b campaign root with baselines/ and rmse_by_init.csv (GPU only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="artefact root (default: outputs/). Use an NFS mount off instance disk.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes logs and tables under <output-dir>/<tag>/",
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


def _apply_cpu_hygiene() -> None:
    """Set the baseline flags that exist on CPU. GPU flags wait for the CUDA path."""
    torch.set_float32_matmul_precision("highest")


def _apply_gpu_hygiene() -> None:
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _sku(device: str) -> str:
    if device != _GPU_DEVICE or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _scalar(value: object) -> float:
    return float(np.asarray(value).item())


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


def _score_variant_vs_baseline(
    variant_preds: list[Batch],
    baseline_preds: list[Batch],
    *,
    init_id: int,
    init_time: datetime,
    spec: ModelSpec,
) -> list[dict[str, Any]]:
    """One init: per-lead RMSE(variant, baseline) as long-form rows."""
    if len(variant_preds) != len(baseline_preds):
        msg = f"variant/baseline step counts differ: {len(variant_preds)} vs {len(baseline_preds)}"
        raise ValueError(msg)
    variables = [*spec.surf_vars, *spec.atmos_vars]
    step_hours = spec.input_timestep_hours
    rows: list[dict[str, Any]] = []
    for step_index, (variant_pred, baseline_pred) in enumerate(
        zip(variant_preds, baseline_preds, strict=True),
        start=1,
    ):
        lead_hours = step_index * step_hours
        rmse = compute_rmse_score_variant_vs_baseline(
            baseline=batch_to_dataset(baseline_pred),
            variant=batch_to_dataset(variant_pred),
            variables=variables,
        )
        lead_rows = _rmse_dataset_to_rows(
            rmse,
            init_id=init_id,
            init_time=init_time,
            lead_hours=lead_hours,
        )
        rows.extend(lead_rows)
        _LOG.info(
            "scored init_id=%s lead=%sh n_rows=%s max_rmse=%.4g",
            init_id,
            lead_hours,
            len(lead_rows),
            max(row["rmse"] for row in lead_rows),
        )
    return rows


def _max_rmse(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return max(float(row["rmse"]) for row in rows)


def _write_session_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cpu_dry_run(*, variant: VariantConfig, output_dir: Path) -> int:
    """fp32-baseline vs itself on AuroraSmallPretrained. Does not call model_factory."""
    _LOG.warning(_NO_SKILL_BANNER)
    _apply_cpu_hygiene()
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
    _LOG.info(
        "loaded %s on %s in %.2fs (not %s)",
        _DRY_RUN_MODEL,
        _DRY_RUN_DEVICE,
        load_s,
        variant.name,
    )

    def _rollout(label: str) -> tuple[list[Batch], float]:
        started = time.perf_counter()
        preds = run_rollout(model, batch, steps=_DRY_RUN_STEPS)
        elapsed_s = time.perf_counter() - started
        _LOG.info("%s rollout wall time: %.2fs for %d steps", label, elapsed_s, _DRY_RUN_STEPS)
        return preds, elapsed_s

    baseline_preds, _ = _rollout("baseline")
    variant_preds, variant_s = _rollout("variant")

    rows = _score_variant_vs_baseline(
        variant_preds,
        baseline_preds,
        init_id=0,
        init_time=_DRY_RUN_INIT,
        spec=spec,
    )
    write_rmse_tables(rows, output_dir)
    max_rmse = _max_rmse(rows)
    _LOG.info(
        "wrote RMSE tables under %s (%s rows, max_rmse=%.4g)",
        output_dir,
        len(rows),
        max_rmse,
    )

    checksums = [
        checksums_for_pred(pred, step=step_index)
        for step_index, pred in enumerate(variant_preds, start=1)
    ]
    run_path = output_dir / "run_init-0.json"
    write_run_log(
        run_path,
        RunLog(
            env=collect_env(),
            run=RunIdentity(
                variant=variant.name,
                init_id=0,
                init_time=_DRY_RUN_INIT.isoformat(),
                sku="cpu",
                device=_DRY_RUN_DEVICE,
                batch_size=1,
                n_steps=_DRY_RUN_STEPS,
                seed=_DRY_RUN_SEED,
                model_name=_DRY_RUN_MODEL,
                offload_to_cpu=False,
                cpu_dry_run=True,
            ),
            timing=TimingRecord(
                model_load_s=load_s,
                full_rollout_wall_s=variant_s,
                full_rollout_ms=None,
                per_step_ms=[None] * _DRY_RUN_STEPS,
            ),
            memory=snapshot_memory(_DRY_RUN_DEVICE),
            checksums=checksums,
            max_rmse_variant_vs_baseline=max_rmse,
        ),
    )
    _LOG.info("wrote run JSON %s", run_path)
    _LOG.warning(_NO_SKILL_BANNER)

    if not np.isfinite(max_rmse) or max_rmse > _RMSE_NEAR_ZERO:
        _LOG.error(
            "dry-run RMSE(variant, baseline) max=%.4g; expected ≈ 0 (atol=%g)",
            max_rmse,
            _RMSE_NEAR_ZERO,
        )
        return 1
    return 0


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


def _log_inference_step(
    label: str,
    step_index: int,
    steps: int,
    *,
    step_hours: int,
    step_s: float,
    rollout_s: float,
) -> None:
    _LOG.info(
        "%s inference step=%s/%s lead=%sh step_wall=%.2fs rollout_wall=%.2fs",
        label,
        step_index,
        steps,
        step_index * step_hours,
        step_s,
        rollout_s,
    )


def _inference_step_logger(
    label: str, steps: int, *, step_hours: int
) -> Callable[[int, Batch], None]:
    started = time.perf_counter()
    last = started

    def on_step(step_index: int, _pred: Batch) -> None:
        nonlocal last
        now = time.perf_counter()
        step_s = now - last
        last = now
        _log_inference_step(
            label,
            step_index,
            steps,
            step_hours=step_hours,
            step_s=step_s,
            rollout_s=now - started,
        )

    return on_step


def _place_model(model: Aurora, device: str) -> Aurora:
    model.eval()
    param = next(model.parameters())
    if str(param.device) != device:
        model = model.to(device)
    return model


def _warmup(model: Aurora, batch: Batch, *, step_hours: int) -> None:
    for index in range(1, _WARMUP_FORWARDS + 1):
        run_rollout(
            model,
            batch,
            steps=1,
            offload_to_cpu=True,
            on_step=_inference_step_logger(
                f"warm-up {index}/{_WARMUP_FORWARDS}", 1, step_hours=step_hours
            ),
        )
        _LOG.info("warm-up forward %s/%s discarded", index, _WARMUP_FORWARDS)


def _log_after_rollout(memory: MemoryRecord, *, persist_init_id: int) -> None:
    rss_mib = memory.peak_rss_bytes / _MIB
    if memory.max_memory_allocated is None:
        _LOG.info(
            "persist_init_id=%s after variant rollout: peak RSS=%.2f MiB (VRAM n/a)",
            persist_init_id,
            rss_mib,
        )
        return
    reserved = 0 if memory.max_memory_reserved is None else memory.max_memory_reserved
    _LOG.info(
        "persist_init_id=%s after variant rollout: peak RSS=%.2f MiB  "
        "VRAM allocated peak=%.2f MiB  VRAM reserved peak=%.2f MiB",
        persist_init_id,
        rss_mib,
        memory.max_memory_allocated / _MIB,
        reserved / _MIB,
    )


def _variant_rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    *,
    persist_init_id: int,
    step_hours: int,
) -> tuple[list[Batch], float, float | None, MemoryRecord]:
    reset_vram_peak(_GPU_DEVICE)
    started = time.perf_counter()
    preds, cuda_ms = measure_cuda_elapsed_ms(
        lambda: run_rollout(
            model,
            batch,
            steps,
            offload_to_cpu=True,
            on_step=_inference_step_logger(
                f"persist_init_id={persist_init_id} variant",
                steps,
                step_hours=step_hours,
            ),
        ),
        device=_GPU_DEVICE,
    )
    wall_s = time.perf_counter() - started
    memory = snapshot_memory(_GPU_DEVICE)
    _log_after_rollout(memory, persist_init_id=persist_init_id)
    return preds, wall_s, cuda_ms, memory


def _score_init_against_archive(
    preds: list[Batch],
    *,
    baseline_dir: Path,
    source: HresT0Source,
    spec: ModelSpec,
    persist_init_id: int,
    init_time: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_hours = spec.input_timestep_hours
    init_dir = init_forecast_dir(baseline_dir, persist_init_id)
    vs_baseline: list[dict[str, Any]] = []
    vs_truth: list[dict[str, Any]] = []
    for step_index, pred in enumerate(preds, start=1):
        lead_hours = step_index * step_hours
        variant_grid = select_persist_fields(batch_to_dataset(pred), FIELDS_HEADLINE)
        baseline_grid = read_lead_forecast(lead_zarr_path(init_dir, lead_hours))
        vs_baseline.extend(
            score_headline_vs_baseline_rows(
                variant_grid,
                baseline_grid,
                init_id=persist_init_id,
                init_time=init_time,
                lead_hours=lead_hours,
            )
        )
        valid_time = init_time + timedelta(hours=lead_hours)
        analysis = analysis_dataset(source, valid_time, spec)
        vs_truth.extend(
            score_headline_vs_analysis_rows(
                variant_grid,
                analysis,
                init_id=persist_init_id,
                init_time=init_time,
                lead_hours=lead_hours,
            )
        )
    return vs_baseline, vs_truth


def _write_init_run_log(
    output_dir: Path,
    *,
    env: Any,
    variant_name: str,
    persist_init_id: int,
    init_time: datetime,
    n_steps: int,
    load_s: float,
    rollout_s: float,
    rollout_ms: float | None,
    preds: list[Batch],
    memory: MemoryRecord,
    max_rmse_vs_baseline: float,
) -> None:
    run_path = output_dir / f"run_init-{persist_init_id}.json"
    write_run_log(
        run_path,
        RunLog(
            env=env,
            run=RunIdentity(
                variant=variant_name,
                init_id=persist_init_id,
                init_time=init_time.isoformat(),
                sku=_sku(_GPU_DEVICE),
                device=_GPU_DEVICE,
                batch_size=1,
                n_steps=n_steps,
                seed=None,
                model_name=variant_name,
                offload_to_cpu=True,
                cpu_dry_run=False,
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
            max_rmse_variant_vs_baseline=max_rmse_vs_baseline,
        ),
    )
    _LOG.info("wrote run JSON %s", run_path)


def _run_gpu_init(
    init_pair: InitPair,
    *,
    persist_init_id: int,
    source: HresT0Source,
    model: Aurora,
    spec: ModelSpec,
    baseline_dir: Path,
    output_dir: Path,
    env: Any,
    variant_name: str,
    load_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    init_time = as_naive_datetime(init_pair.init_time)
    batch = source.load(init_time, spec)
    validate_batch(batch, spec)
    batch = batch.to(_GPU_DEVICE)
    _LOG.info(
        "persist_init_id=%s init_time=%s steps=%s",
        persist_init_id,
        init_time.isoformat(),
        init_pair.n_rollout_steps,
    )
    preds, wall_s, cuda_ms, memory = _variant_rollout(
        model,
        batch,
        init_pair.n_rollout_steps,
        persist_init_id=persist_init_id,
        step_hours=spec.input_timestep_hours,
    )
    del batch
    vs_baseline, vs_truth = _score_init_against_archive(
        preds,
        baseline_dir=baseline_dir,
        source=source,
        spec=spec,
        persist_init_id=persist_init_id,
        init_time=init_time,
    )
    max_vs_baseline = _max_rmse(vs_baseline)
    _LOG.info(
        "persist_init_id=%s scored vs_baseline=%s rows max=%.4g; vs_truth=%s rows",
        persist_init_id,
        len(vs_baseline),
        max_vs_baseline,
        len(vs_truth),
    )
    _write_init_run_log(
        output_dir,
        env=env,
        variant_name=variant_name,
        persist_init_id=persist_init_id,
        init_time=init_time,
        n_steps=init_pair.n_rollout_steps,
        load_s=load_s,
        rollout_s=wall_s,
        rollout_ms=cuda_ms,
        preds=preds,
        memory=memory,
        max_rmse_vs_baseline=max_vs_baseline,
    )
    del preds
    return vs_baseline, vs_truth


def _run_gpu_campaign(
    *,
    variant: VariantConfig,
    baseline_dir: Path,
    output_dir: Path,
    rollout_config: Path,
    max_inits: int | None,
) -> int:
    _apply_gpu_hygiene()
    spec = variant.spec
    try:
        schedule = _load_campaign(rollout_config)
        persist_ids = persist_init_ids_by_time(baseline_dir)
    except (FileNotFoundError, SpliceCoverageError, TypeError, ValueError) as exc:
        _LOG.error("%s", exc)
        return 1

    source = _open_hres_source(schedule.config.splice_path)
    load_started = time.perf_counter()
    model = _place_model(variant.model_factory(), _GPU_DEVICE)
    load_s = time.perf_counter() - load_started
    _LOG.info("loaded variant %s on %s in %.2fs", variant.name, _GPU_DEVICE, load_s)
    env = collect_env()

    init_pairs = list(schedule.init_pairs)
    if max_inits is not None:
        init_pairs = init_pairs[:max_inits]
        _LOG.info("capping campaign at %s inits (of %s)", len(init_pairs), len(schedule.init_pairs))
    if not init_pairs:
        _LOG.error("campaign has no inits")
        return 1

    resolved: list[tuple[InitPair, int]] = []
    for init_pair in init_pairs:
        init_time = as_naive_datetime(init_pair.init_time)
        persist_init_id = persist_ids.get(init_time.isoformat())
        if persist_init_id is None:
            _LOG.error(
                "no persist init_id for %s in %s/rmse_by_init.csv",
                init_time.isoformat(),
                baseline_dir,
            )
            return 1
        resolved.append((init_pair, persist_init_id))

    first_pair = resolved[0][0]
    first_init = as_naive_datetime(first_pair.init_time)
    batch = source.load(first_init, spec)
    validate_batch(batch, spec)
    batch = batch.to(_GPU_DEVICE)
    _warmup(model, batch, step_hours=spec.input_timestep_hours)
    del batch

    session_path = output_dir / "session.json"
    _write_session_json(
        session_path,
        {
            "baseline_dir": str(baseline_dir),
            "cpu_dry_run": False,
            "device": _GPU_DEVICE,
            "env": env.model_dump(mode="json"),
            "fields": FIELDS_HEADLINE,
            "n_steps": first_pair.n_rollout_steps,
            "rollout_config": str(rollout_config),
            "sku": _sku(_GPU_DEVICE),
            "variant": variant.name,
            "warmup_forwards": _WARMUP_FORWARDS,
        },
    )
    _LOG.info("wrote session env %s", session_path)

    vs_baseline_dir = output_dir / _VS_BASELINE
    vs_truth_dir = output_dir / _VS_TRUTH
    vs_baseline_rows = load_rmse_rows(vs_baseline_dir)
    vs_truth_rows = load_rmse_rows(vs_truth_dir)
    done_ids = {int(row["init_id"]) for row in vs_baseline_rows}

    for init_pair, persist_init_id in resolved:
        if persist_init_id in done_ids:
            _LOG.info("skip persist_init_id=%s (already scored)", persist_init_id)
            continue
        scored_baseline, scored_truth = _run_gpu_init(
            init_pair,
            persist_init_id=persist_init_id,
            source=source,
            model=model,
            spec=spec,
            baseline_dir=baseline_dir,
            output_dir=output_dir,
            env=env,
            variant_name=variant.name,
            load_s=load_s,
        )
        vs_baseline_rows.extend(scored_baseline)
        vs_truth_rows.extend(scored_truth)
        write_rmse_tables(vs_baseline_rows, vs_baseline_dir)
        write_rmse_tables(vs_truth_rows, vs_truth_dir)
        done_ids.add(persist_init_id)
        _LOG.info(
            "wrote RMSE tables vs_baseline=%s vs_truth=%s",
            len(vs_baseline_rows),
            len(vs_truth_rows),
        )

    if not vs_baseline_rows or not vs_truth_rows:
        _LOG.error("no RMSE rows produced")
        return 1
    write_rmse_tables(vs_baseline_rows, vs_baseline_dir)
    write_rmse_tables(vs_truth_rows, vs_truth_dir)
    _LOG.info(_TEARDOWN_REMINDER)
    return 0


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

    variant = _lookup_variant(args.variant)
    if variant is None:
        known = ", ".join(sorted(VARIANTS))
        print(f"unknown variant {args.variant!r}; known: {known}", file=sys.stderr)
        return 2

    if args.cpu_dry_run:
        if args.rollout_config is not None:
            print(
                "--cpu-dry-run does not take --rollout-config (uses SyntheticSource)",
                file=sys.stderr,
            )
            return 2
        if args.baseline_dir is not None:
            print("--cpu-dry-run does not take --baseline-dir", file=sys.stderr)
            return 2
        if variant.name != "fp32-baseline":
            print(
                "cpu-dry-run scores fp32-baseline against itself; other variants are GPU work",
                file=sys.stderr,
            )
            return 2
        output_dir = run_artifact_dir(args.output_dir.expanduser(), tag)
        configure_run_logging(output_dir / "run_fidelity.log", tag=tag)
        _LOG.info("Q2 fidelity CPU dry-run; log file: %s", output_dir / "run_fidelity.log")
        return _run_cpu_dry_run(variant=variant, output_dir=output_dir)

    if args.baseline_dir is None:
        print("--baseline-dir is required for the GPU campaign", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print(
            "CUDA is required for the GPU campaign. Pass --cpu-dry-run for the wiring check.",
            file=sys.stderr,
        )
        return 2

    rollout_config = _DEFAULT_GPU_ROLLOUT if args.rollout_config is None else args.rollout_config
    output_dir = run_artifact_dir(args.output_dir.expanduser(), tag)
    configure_run_logging(output_dir / "run_fidelity.log", tag=tag)
    _LOG.info("Q2 fidelity GPU campaign; log file: %s", output_dir / "run_fidelity.log")
    return _run_gpu_campaign(
        variant=variant,
        baseline_dir=args.baseline_dir.expanduser(),
        output_dir=output_dir,
        rollout_config=rollout_config,
        max_inits=args.max_inits,
    )


if __name__ == "__main__":
    raise SystemExit(main())
