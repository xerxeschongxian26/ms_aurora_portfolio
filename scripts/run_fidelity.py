"""Q2 fidelity campaign: named variant vs a baseline forecast, one rollout TOML.

WP4 first draft — CPU dry-run only. Scores ``fp32-baseline`` against itself on
``AuroraSmallPretrained`` (synthetic 32×64 batch, 1 init, 2 steps). That checks
the seam: registry lookup → rollout → ``batch_to_dataset`` → RMSE helpers →
tables + a stub run JSON. RMSE(variant, baseline) must be ≈ 0.

This is plumbing, not skill. Do not quote these RMSEs. The GPU path (persisted
baseline forecasts, screen/confirm TOML, ``runlog.py``) is not in this draft.

Example::

    uv sync --extra forecast
    uv run --extra forecast python scripts/run_fidelity.py --cpu-dry-run --tag wiring
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from aurora import Aurora, Batch

from aurora_inference.contract import ModelSpec, validate_batch
from aurora_inference.data.synthetic import SyntheticSource
from aurora_inference.evaluation.fidelity import compute_rmse_score_variant_vs_baseline
from aurora_inference.evaluation.grids import batch_to_dataset
from aurora_inference.evaluation.tables import write_rmse_tables
from aurora_inference.evaluation.variants import VARIANTS, VariantConfig
from aurora_inference.inference.forward import run_rollout
from aurora_inference.logging import configure_run_logging, normalize_run_tag, run_artifact_dir
from aurora_inference.model.loader import load_model

_LOG = logging.getLogger(__name__)

_OUTPUT_DIR = Path("outputs")
_DRY_RUN_DEVICE = "cpu"
_DRY_RUN_MODEL = "aurora-small-pretrained"
_DRY_RUN_STEPS = 2
_DRY_RUN_INIT = datetime(2022, 1, 1, 12, 0)
_DRY_RUN_HEIGHT = 32
_DRY_RUN_WIDTH = 64
_DRY_RUN_SEED = 42
_RMSE_NEAR_ZERO = 1e-5

_NO_SKILL_BANNER = (
    "=== NO FORECAST SKILL === CPU dry-run uses AuroraSmallPretrained + "
    "SyntheticSource. RMSE≈0 only proves the Q2 wiring, not forecast quality."
)

# TODO(stage-2): replace this stub with aurora_inference.runlog — blocked on WP4
# runlog.py (CUDA Event timing, torch.backends.* readback, reserved VRAM).
_RUNLOG_STUB_NOTE = "stub JSON; replace with aurora_inference.runlog once runlog.py lands"


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
        help="GPU campaign TOML (not implemented in this draft)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes logs and tables under outputs/<tag>/",
    )
    return parser.parse_args(argv)


def _lookup_variant(name: str) -> VariantConfig | None:
    return VARIANTS.get(name)


def _apply_cpu_hygiene() -> None:
    """Set the baseline flags that exist on CPU. GPU flags wait for the CUDA path."""
    torch.set_float32_matmul_precision("highest")


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


def _checksums_for_pred(pred: Batch) -> dict[str, float]:
    """Cheap z500 / t850 witnesses without storing tensors."""
    levels = [int(level) for level in pred.metadata.atmos_levels]
    i500 = levels.index(500)
    i850 = levels.index(850)
    z500 = pred.atmos_vars["z"][0, 0, i500]
    t850 = pred.atmos_vars["t"][0, 0, i850]
    return {
        "z500_sum": float(z500.sum()),
        "z500_abs_max": float(z500.abs().max()),
        "t850_sum": float(t850.sum()),
        "t850_abs_max": float(t850.abs().max()),
    }


def _write_stub_run_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _LOG.info("wrote stub run JSON %s", path)


def _max_rmse(rows: list[dict[str, Any]]) -> float:
    return max(float(row["rmse"]) for row in rows)


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

    baseline_preds, baseline_s = _rollout("baseline")
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
        {"step": step_index, **_checksums_for_pred(pred)}
        for step_index, pred in enumerate(variant_preds, start=1)
    ]
    _write_stub_run_json(
        output_dir / "run_init-0.json",
        {
            "note": _RUNLOG_STUB_NOTE,
            "cpu_dry_run": True,
            "variant": variant.name,
            "init_id": 0,
            "init_time": _DRY_RUN_INIT.isoformat(),
            "sku": "cpu",
            "batch_size": 1,
            "n_steps": _DRY_RUN_STEPS,
            "seed": _DRY_RUN_SEED,
            "model_name": _DRY_RUN_MODEL,
            "timing": {
                "model_load_s": load_s,
                "baseline_rollout_s": baseline_s,
                "variant_rollout_s": variant_s,
            },
            "checksums": checksums,
            "max_rmse_variant_vs_baseline": max_rmse,
        },
    )
    _LOG.warning(_NO_SKILL_BANNER)

    if not np.isfinite(max_rmse) or max_rmse > _RMSE_NEAR_ZERO:
        _LOG.error(
            "dry-run RMSE(variant, baseline) max=%.4g; expected ≈ 0 (atol=%g)",
            max_rmse,
            _RMSE_NEAR_ZERO,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tag = normalize_run_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    variant = _lookup_variant(args.variant)
    if variant is None:
        known = ", ".join(sorted(VARIANTS))
        print(f"unknown variant {args.variant!r}; known: {known}", file=sys.stderr)
        return 2

    if not args.cpu_dry_run:
        print(
            "GPU fidelity campaign is not in this draft "
            "(needs runlog.py and persisted baseline forecasts). "
            "Pass --cpu-dry-run for the WP4 wiring check.",
            file=sys.stderr,
        )
        return 2
    if args.rollout_config is not None:
        print(
            "--cpu-dry-run does not take --rollout-config (uses SyntheticSource)",
            file=sys.stderr,
        )
        return 2
    if variant.name != "fp32-baseline":
        print(
            "cpu-dry-run scores fp32-baseline against itself; other variants are Stage 3 GPU work",
            file=sys.stderr,
        )
        return 2

    output_dir = run_artifact_dir(_OUTPUT_DIR, tag)
    configure_run_logging(output_dir / "run_fidelity.log", tag=tag)
    _LOG.info("Q2 fidelity CPU dry-run; log file: %s", output_dir / "run_fidelity.log")
    return _run_cpu_dry_run(variant=variant, output_dir=output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
