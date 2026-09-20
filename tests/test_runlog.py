"""Run JSON schema, hygiene readback, and CPU-safe CUDA Event skip (no weights)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import make_valid_batch

from aurora_inference.config import AURORA_HF_REVISION
from aurora_inference.runlog import (
    DeclaredPrecision,
    EnvSnapshot,
    HygieneFlags,
    MemoryRecord,
    ObservedModuleDType,
    RunIdentity,
    RunLog,
    StepChecksum,
    TimingRecord,
    checksums_for_pred,
    collect_env,
    cuda_timing_available,
    load_run_log,
    measure_cuda_elapsed_ms,
    read_attention_routine,
    read_hygiene_flags,
    reset_vram_peak,
    snapshot_memory,
    write_run_log,
)


def _sample_run_log() -> RunLog:
    return RunLog(
        env=EnvSnapshot(
            gpu_name=None,
            driver_version=None,
            cuda_runtime=None,
            torch_version="2.0.0",
            git_sha="abc123",
            aurora_hf_revision=AURORA_HF_REVISION,
            pip_freeze_path=None,
            hygiene=HygieneFlags(
                float32_matmul_precision="highest",
                cudnn_benchmark=False,
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=False,
                inference_mode_enabled=False,
            ),
        ),
        run=RunIdentity(
            variant="fp32-baseline",
            init_id=0,
            init_time="2022-01-01T12:00:00",
            sku="cpu",
            device="cpu",
            batch_size=1,
            n_steps=2,
            seed=42,
            model_name="aurora-small-pretrained",
            offload_to_cpu=False,
            cpu_dry_run=True,
        ),
        timing=TimingRecord(
            model_load_s=1.5,
            full_rollout_wall_s=0.2,
            full_rollout_ms=None,
            per_step_ms=[None, None],
        ),
        memory=MemoryRecord(
            peak_rss_bytes=1024,
            max_memory_allocated=None,
            max_memory_reserved=None,
        ),
        checksums=[
            StepChecksum(
                step=1,
                z500_sum=0.0,
                z500_abs_max=0.0,
                t850_sum=0.0,
                t850_abs_max=0.0,
            )
        ],
        warnings=[],
        max_rmse_variant_vs_baseline=0.0,
    )


def test_run_log_json_round_trips(tmp_path: Path) -> None:
    original = _sample_run_log()
    path = tmp_path / "run_init-0.json"
    write_run_log(path, original)
    loaded = load_run_log(path)
    assert loaded == original


def test_cuda_timing_skips_events_on_cpu() -> None:
    assert cuda_timing_available("cpu") is False
    result, elapsed_ms = measure_cuda_elapsed_ms(lambda: 7, device="cpu")
    assert result == 7
    assert elapsed_ms is None


def test_read_hygiene_flags_matches_torch_state() -> None:
    torch.set_float32_matmul_precision("highest")
    flags = read_hygiene_flags()
    assert flags.float32_matmul_precision == "highest"
    assert flags.inference_mode_enabled is False
    with torch.inference_mode():
        inside = read_hygiene_flags()
    assert inside.inference_mode_enabled is True


def test_collect_env_records_pin_and_torch_version() -> None:
    env = collect_env()
    assert env.aurora_hf_revision == AURORA_HF_REVISION
    assert env.torch_version == torch.__version__
    assert env.hygiene.float32_matmul_precision == torch.get_float32_matmul_precision()


def test_snapshot_memory_cpu_has_no_vram() -> None:
    reset_vram_peak("cpu")
    memory = snapshot_memory("cpu")
    assert memory.peak_rss_bytes > 0
    assert memory.max_memory_allocated is None
    assert memory.max_memory_reserved is None


def test_checksums_for_pred_reads_z500_and_t850() -> None:
    batch = make_valid_batch()
    levels = [int(level) for level in batch.metadata.atmos_levels]
    i500 = levels.index(500)
    batch.atmos_vars["z"][0, -1, i500] = 3.0
    checksum = checksums_for_pred(batch, step=1)
    expected_sum = 3.0 * batch.atmos_vars["z"][0, -1, i500].numel()
    assert checksum.step == 1
    assert checksum.z500_sum == pytest.approx(expected_sum)
    assert checksum.z500_abs_max == pytest.approx(3.0)


def test_load_run_log_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"not_a_run_log": true}\n', encoding="utf-8")
    with pytest.raises(Exception, match="env|Field|required"):
        load_run_log(path)


def test_run_log_round_trips_precision_fields_without_gpu(tmp_path: Path) -> None:
    original = _sample_run_log()
    original = original.model_copy(
        update={
            "declared_precision": DeclaredPrecision(
                weights_dtype="bf16",
                matmul_operand="bf16",
                accumulator="unguaranteed",
                reduction_ops="ambient",
                scope="whole-model",
            ),
            "observed_formats": [
                ObservedModuleDType(
                    module="encoder",
                    input_dtype="bf16",
                    output_dtype="bf16",
                    qualified_name="encoder",
                )
            ],
            "declared_vs_observed": "PASS",
            "attention_routine": None,
            "finiteness_passed": True,
            "inverted_zero_verdict": "SUSPECT — variant may not have applied",
        }
    )
    path = tmp_path / "run_init-0.json"
    write_run_log(path, original)
    loaded = load_run_log(path)
    assert loaded == original
    if torch.cuda.is_available():
        assert isinstance(read_attention_routine(), str)
    else:
        assert read_attention_routine() is None
    flags = read_hygiene_flags()
    assert flags.allow_fp16_reduced_precision_reduction is None or isinstance(
        flags.allow_fp16_reduced_precision_reduction, bool
    )
    assert flags.allow_bf16_reduced_precision_reduction is None or isinstance(
        flags.allow_bf16_reduced_precision_reduction, bool
    )
