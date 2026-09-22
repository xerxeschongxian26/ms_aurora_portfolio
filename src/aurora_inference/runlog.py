"""Structured per-run JSON for Q2 / WP5b (env, flags, timing, memory, checksums).

Not the human INFO stream (:mod:`aurora_inference.logging`) and not the profiler
trace (WP5b D9 ``trace.json``). One JSON per init; env is included so a file is
self-contained after the box is gone.

Hygiene flags are read back from ``torch.backends.*`` / ``torch.get_*``, not
copied from ``variants.py`` intent. CUDA ``Event`` timing is a no-op on CPU
(returns ``None`` ms). Power sidecar is out of scope for this draft.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import torch
from aurora import Batch
from pydantic import BaseModel, ConfigDict, Field

from aurora_inference.config import AURORA_HF_REVISION
from aurora_inference.logging import peak_rss_bytes

__all__ = [
    "DeclaredPrecision",
    "EnvSnapshot",
    "HygieneFlags",
    "MemoryRecord",
    "ObservedModuleDType",
    "RunIdentity",
    "RunLog",
    "StepChecksum",
    "TimingRecord",
    "checksums_for_pred",
    "collect_env",
    "cuda_timing_available",
    "load_run_log",
    "measure_cuda_elapsed_ms",
    "read_attention_routine",
    "read_hygiene_flags",
    "reset_vram_peak",
    "snapshot_memory",
    "write_run_log",
]

_NVIDIA_SMI_TIMEOUT_S = 5.0
_GIT_TIMEOUT_S = 5.0


class HygieneFlags(BaseModel):
    """Torch flags as actually set at snapshot time — not the variant description."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    float32_matmul_precision: str
    cudnn_benchmark: bool | None
    cuda_matmul_allow_tf32: bool | None
    cudnn_allow_tf32: bool | None
    inference_mode_enabled: bool
    allow_fp16_reduced_precision_reduction: bool | None = None
    allow_bf16_reduced_precision_reduction: bool | None = None


class EnvSnapshot(BaseModel):
    """Host / framework identity for one session (copied into each run JSON)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_name: str | None
    driver_version: str | None
    cuda_runtime: str | None
    torch_version: str
    git_sha: str | None
    aurora_hf_revision: str
    pip_freeze_path: str | None
    hygiene: HygieneFlags


class RunIdentity(BaseModel):
    """What was run — caller-supplied, not inferred from env."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    init_id: int
    init_time: str | None = None
    sku: str
    device: str
    batch_size: int = Field(ge=1)
    n_steps: int = Field(ge=1)
    seed: int | None = None
    model_name: str | None = None
    offload_to_cpu: bool = False
    cpu_dry_run: bool = False


class TimingRecord(BaseModel):
    """Load wall uses ``perf_counter``. CUDA step times use ``Event`` (None on CPU)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_load_s: float | None = None
    full_rollout_wall_s: float | None = None
    full_rollout_ms: float | None = None
    per_step_ms: list[float | None] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    """Host RSS plus CUDA allocated / reserved peaks (None when not CUDA)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    peak_rss_bytes: int
    max_memory_allocated: int | None = None
    max_memory_reserved: int | None = None


class StepChecksum(BaseModel):
    """Cheap z500 / t850 witnesses without storing tensors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=1)
    z500_sum: float
    z500_abs_max: float
    t850_sum: float
    t850_abs_max: float


class ObservedModuleDType(BaseModel):
    """Number format entering and leaving one hooked module, once per run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    input_dtype: str
    output_dtype: str
    qualified_name: str | None = None


class DeclaredPrecision(BaseModel):
    """The five typed fields copied from ``VariantConfig`` at run time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights_dtype: str
    matmul_operand: str
    accumulator: str
    reduction_ops: str
    scope: str


class RunLog(BaseModel):
    """One inference run: env + identity + timing + memory + checksums."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env: EnvSnapshot
    run: RunIdentity
    timing: TimingRecord
    memory: MemoryRecord
    checksums: list[StepChecksum] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    max_rmse_variant_vs_baseline: float | None = None
    declared_precision: DeclaredPrecision | None = None
    observed_formats: list[ObservedModuleDType] = Field(default_factory=list)
    declared_vs_observed: str | None = None
    attention_routine: str | None = None
    finiteness_passed: bool | None = None
    inverted_zero_verdict: str | None = None


def read_hygiene_flags() -> HygieneFlags:
    """Read matmul / cuDNN flags from torch — do not copy variant intent."""
    return HygieneFlags(
        float32_matmul_precision=str(torch.get_float32_matmul_precision()),
        cudnn_benchmark=_optional_bool(lambda: torch.backends.cudnn.benchmark),
        cuda_matmul_allow_tf32=_optional_bool(lambda: torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=_optional_bool(lambda: torch.backends.cudnn.allow_tf32),
        inference_mode_enabled=bool(torch.is_inference_mode_enabled()),
        allow_fp16_reduced_precision_reduction=_optional_bool(
            lambda: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        allow_bf16_reduced_precision_reduction=_optional_bool(
            lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    )


def read_attention_routine() -> str | None:
    """Enabled SDPA backends. None on a machine with no CUDA.

    The kernel picked per call is not exposed in Python; this is the observable
    allow-list. Do not change which routine is used — ``TODO(stage-3-followup)``.
    """
    if not torch.cuda.is_available():
        return None
    enabled: list[str] = []
    for name, reader in (
        ("flash", torch.backends.cuda.flash_sdp_enabled),
        ("mem_efficient", torch.backends.cuda.mem_efficient_sdp_enabled),
        ("math", torch.backends.cuda.math_sdp_enabled),
        ("cudnn", torch.backends.cuda.cudnn_sdp_enabled),
    ):
        if _optional_bool(reader):
            enabled.append(name)
    return ",".join(enabled) if enabled else "unknown"


def collect_env(*, pip_freeze_path: Path | str | None = None) -> EnvSnapshot:
    """Snapshot GPU / torch / git / checkpoint pin. nvidia-smi is best-effort."""
    cuda_available = torch.cuda.is_available()
    gpu_name: str | None = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if gpu_name is None:
        gpu_name = _nvidia_smi_field("name")
    freeze = None if pip_freeze_path is None else str(pip_freeze_path)
    return EnvSnapshot(
        gpu_name=gpu_name,
        driver_version=_nvidia_smi_field("driver_version"),
        cuda_runtime=torch.version.cuda,
        torch_version=torch.__version__,
        git_sha=_git_sha(),
        aurora_hf_revision=AURORA_HF_REVISION,
        pip_freeze_path=freeze,
        hygiene=read_hygiene_flags(),
    )


def cuda_timing_available(device: str | torch.device) -> bool:
    """True only when CUDA Events can record elapsed time."""
    torch_device = torch.device(device)
    return torch_device.type == "cuda" and torch.cuda.is_available()


def measure_cuda_elapsed_ms[T](
    fn: Callable[[], T],
    *,
    device: str | torch.device,
) -> tuple[T, float | None]:
    """Run ``fn`` and return ``(result, elapsed_ms)``. ``elapsed_ms`` is None on CPU.

    Does not construct ``torch.cuda.Event`` unless CUDA timing is available.
    """
    if not cuda_timing_available(device):
        return fn(), None
    torch_device = torch.device(device)
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize(torch_device)
    return result, float(start.elapsed_time(end))


def reset_vram_peak(device: str | torch.device) -> None:
    """Reset the CUDA allocation high-water mark. No-op if not CUDA."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(torch_device)
    torch.cuda.reset_peak_memory_stats(_cuda_device_index(torch_device))


def snapshot_memory(device: str | torch.device) -> MemoryRecord:
    """Peak RSS plus CUDA allocated / reserved peaks since the last reset."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return MemoryRecord(peak_rss_bytes=peak_rss_bytes())
    torch.cuda.synchronize(torch_device)
    index = _cuda_device_index(torch_device)
    return MemoryRecord(
        peak_rss_bytes=peak_rss_bytes(),
        max_memory_allocated=int(torch.cuda.max_memory_allocated(index)),
        max_memory_reserved=int(torch.cuda.max_memory_reserved(index)),
    )


def checksums_for_pred(pred: Batch, *, step: int) -> StepChecksum:
    """z500 / t850 sum and abs-max on the last (T=1) time index."""
    if step < 1:
        msg = f"step must be >= 1 (got {step})"
        raise ValueError(msg)
    levels = [int(level) for level in pred.metadata.atmos_levels]
    try:
        i500 = levels.index(500)
        i850 = levels.index(850)
    except ValueError as exc:
        msg = f"pred atmos_levels must include 500 and 850 (got {levels})"
        raise ValueError(msg) from exc
    z500 = pred.atmos_vars["z"][0, -1, i500]
    t850 = pred.atmos_vars["t"][0, -1, i850]
    return StepChecksum(
        step=step,
        z500_sum=float(z500.sum()),
        z500_abs_max=float(z500.abs().max()),
        t850_sum=float(t850.sum()),
        t850_abs_max=float(t850.abs().max()),
    )


def write_run_log(path: Path, run_log: RunLog) -> None:
    """Write one run JSON (sorted keys, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run_log.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_run_log(path: Path) -> RunLog:
    """Load and validate a run JSON against :class:`RunLog`."""
    return RunLog.model_validate_json(path.read_text(encoding="utf-8"))


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _optional_bool(read: Callable[[], object]) -> bool | None:
    try:
        value = read()
    except (AttributeError, RuntimeError, AssertionError):
        return None
    if value is None:
        return None
    return bool(value)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _nvidia_smi_field(query: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    value = lines[0]
    if value.lower() in {"", "[n/a]", "n/a"}:
        return None
    return value
