"""Laptop dry-run of Stage 3 precision variants on the small debug model.

Not ``--cpu-dry-run``: that path still scores ``fp32-baseline`` against itself
and never calls ``model_factory``. This module applies the same helpers the
registered factories use to untrained ``AuroraSmallPretrained``. Plumbing only
— no forecast skill. CUDA-only AMP rows must refuse rather than run FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from aurora import Batch

from aurora_inference.contract import validate_batch
from aurora_inference.data.synthetic import SyntheticSource
from aurora_inference.evaluation.precision import (
    DeclaredObservedMismatchError,
    check_declared_versus_observed,
    declared_precision,
    install_dtype_hooks,
)
from aurora_inference.evaluation.variants import (
    CUDA_ONLY_VARIANTS,
    VARIANTS,
    VariantConfig,
    build_debug_variant,
)
from aurora_inference.inference.forward import run_rollout
from aurora_inference.runlog import (
    RunIdentity,
    RunLog,
    TimingRecord,
    checksums_for_pred,
    collect_env,
    snapshot_memory,
    write_run_log,
)

__all__ = [
    "CUDA_ONLY_REFUSE_MESSAGE",
    "DEBUG_DRY_RUN_STEPS",
    "DebugVariantResult",
    "dry_run_debug_variant",
    "lookup_precision_variant",
]

CUDA_ONLY_REFUSE_MESSAGE = "whole-forward autocast requires CUDA; refusing to silently run in FP32"
DEBUG_DRY_RUN_STEPS = 2
_DEVICE = "cpu"
_DEBUG_MODEL = "aurora-small-pretrained"
_DRY_RUN_INIT = datetime(2022, 1, 1, 12, 0)
_DRY_RUN_HEIGHT = 32
_DRY_RUN_WIDTH = 64
_DRY_RUN_SEED = 42
_TF32_CPU_NOTE = (
    "tf32-matmul set float32_matmul_precision='high'; TF32 kernels are CUDA Ampere "
    "and do not run on CPU. Flag is set; arithmetic stays host FP32."
)


@dataclass(frozen=True)
class DebugVariantResult:
    """One variant's laptop dry-run: ran, or refused with a recorded reason."""

    variant: str
    outcome: Literal["ran", "refused"]
    reason: str | None
    declared_vs_observed: str
    run_log_path: Path


def dry_run_debug_variant(variant: VariantConfig, output_dir: Path) -> DebugVariantResult:
    """Construct, hook, and roll two synthetic steps (or refuse) on CPU.

    Untrained debug weights emit NaNs, so finite checks are off. DVO still runs
    after a completed forward. CUDA-only AMP raises before a silent FP32 pass.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"run_{variant.name}.json"
    source = SyntheticSource(height=_DRY_RUN_HEIGHT, width=_DRY_RUN_WIDTH, seed=_DRY_RUN_SEED)
    batch = source.load(_DRY_RUN_INIT, variant.spec)
    validate_batch(batch, variant.spec)
    batch = batch.to(_DEVICE)
    warnings: list[str] = []
    if variant.name == "tf32-matmul" and not torch.cuda.is_available():
        warnings.append(_TF32_CPU_NOTE)

    model = build_debug_variant(variant.name)
    hooks = install_dtype_hooks(model)
    preds: list[Batch] = []
    dvo = "not-run"
    outcome: Literal["ran", "refused"] = "refused"
    reason: str | None = None
    try:
        preds = run_rollout(
            model,
            batch,
            steps=DEBUG_DRY_RUN_STEPS,
            init_time=_DRY_RUN_INIT,
            check_finite=False,
        )
        try:
            dvo = check_declared_versus_observed(variant, model, hooks.observations)
        except DeclaredObservedMismatchError as exc:
            dvo = str(exc)
            warnings.append(dvo)
        outcome = "ran"
    except RuntimeError as exc:
        reason = str(exc)
        dvo = f"REFUSED — {reason}"
        warnings.append(reason)
        outcome = "refused"
        if variant.name in CUDA_ONLY_VARIANTS and CUDA_ONLY_REFUSE_MESSAGE not in reason:
            msg = f"{variant.name} refused, but not with the CUDA-only AMP message (got {reason!r})"
            raise RuntimeError(msg) from exc
    finally:
        observed = list(hooks.observations)
        hooks.remove()

    write_run_log(
        run_path,
        RunLog(
            env=collect_env(),
            run=RunIdentity(
                variant=variant.name,
                init_id=0,
                init_time=_DRY_RUN_INIT.isoformat(),
                sku="cpu",
                device=_DEVICE,
                batch_size=1,
                n_steps=DEBUG_DRY_RUN_STEPS,
                seed=_DRY_RUN_SEED,
                model_name=_DEBUG_MODEL,
                offload_to_cpu=False,
                cpu_dry_run=False,
            ),
            timing=TimingRecord(
                model_load_s=None,
                full_rollout_wall_s=None,
                full_rollout_ms=None,
                per_step_ms=[None] * DEBUG_DRY_RUN_STEPS,
            ),
            memory=snapshot_memory(_DEVICE),
            checksums=[
                checksums_for_pred(pred, step=step_index)
                for step_index, pred in enumerate(preds, start=1)
            ],
            warnings=warnings,
            max_rmse_variant_vs_baseline=None,
            declared_precision=declared_precision(variant),
            observed_formats=observed,
            declared_vs_observed=dvo,
            attention_routine=None,
            finiteness_passed=None,
            inverted_zero_verdict=None,
        ),
    )
    return DebugVariantResult(
        variant=variant.name,
        outcome=outcome,
        reason=reason,
        declared_vs_observed=dvo,
        run_log_path=run_path,
    )


def lookup_precision_variant(name: str) -> VariantConfig:
    """Resolve a precision-variant name. Reject ``fp32-baseline`` (not WP2.1)."""
    if name == "fp32-baseline":
        msg = "WP2.1 dry-runs the six precision rows; fp32-baseline is not in that set"
        raise KeyError(msg)
    variant = VARIANTS.get(name)
    if variant is None:
        known = ", ".join(sorted(VARIANTS))
        msg = f"unknown variant {name!r}; known: {known}"
        raise KeyError(msg)
    return variant
