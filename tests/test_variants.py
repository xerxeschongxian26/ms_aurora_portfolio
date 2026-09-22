"""Precision-variant registry (no HuggingFace, no finetuned weights)."""

from __future__ import annotations

import pytest
import torch
from aurora import AuroraSmallPretrained
from conftest import make_valid_batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.evaluation.variants import (
    CUDA_ONLY_VARIANTS,
    PRECISION_VARIANT_NAMES,
    VARIANTS,
    VariantConfig,
    _cast_weather_tensors,
    _convert_incoming_batch_to_param_dtype,
    build_debug_variant,
)
from aurora_inference.inference.forward import run_rollout

_EXPECTED_FIELDS: dict[str, tuple[str, str, str, str, str]] = {
    "fp32-baseline": ("fp32", "fp32", "fp32", "fp32-pinned", "whole-model"),
    "tf32-matmul": ("fp32", "tf32", "fp32", "fp32-pinned", "matmul-only"),
    "bf16-amp-backbone": ("fp32", "bf16", "fp32", "fp32-pinned", "backbone-only"),
    "bf16-amp-full": ("fp32", "bf16", "fp32", "fp32-pinned", "whole-model"),
    "bf16-weights": ("bf16", "bf16", "unguaranteed", "ambient", "whole-model"),
    "fp16-weights": ("fp16", "fp16", "unguaranteed", "ambient", "whole-model"),
    "fp16-weights-amp": ("fp16", "fp16", "unguaranteed", "fp32-pinned-partial", "whole-model"),
}


def test_registered_variants_match_typed_field_table() -> None:
    assert set(VARIANTS) == set(_EXPECTED_FIELDS)
    for name, expected in _EXPECTED_FIELDS.items():
        variant = VARIANTS[name]
        assert variant.name == name
        assert variant.spec == AURORA_PRETRAINED_SPEC
        assert (
            variant.weights_dtype,
            variant.matmul_operand,
            variant.accumulator,
            variant.reduction_ops,
            variant.scope,
        ) == expected


def test_fp32_baseline_name_is_unchanged() -> None:
    assert "fp32-baseline" in VARIANTS
    assert VARIANTS["fp32-baseline"].name == "fp32-baseline"


def test_bf16_weights_description_keeps_upstream_flag_searchable() -> None:
    assert "bf16_mode=True" in VARIANTS["bf16-weights"].description


def test_non_baseline_descriptions_do_not_copy_highest_matmul() -> None:
    for name, variant in VARIANTS.items():
        if name == "fp32-baseline":
            continue
        assert "highest" not in variant.description


def test_typed_fields_are_required() -> None:
    with pytest.raises(TypeError):
        VariantConfig(  # type: ignore[call-arg]
            name="missing-fields",
            model_factory=VARIANTS["fp32-baseline"].model_factory,
            spec=AURORA_PRETRAINED_SPEC,
            description="prose only",
        )


def test_weight_converted_config_rejects_protected_accumulator() -> None:
    with pytest.raises(ValueError, match="unguaranteed"):
        VariantConfig(
            name="bad-bf16",
            model_factory=VARIANTS["fp32-baseline"].model_factory,
            spec=AURORA_PRETRAINED_SPEC,
            weights_dtype="bf16",
            matmul_operand="bf16",
            accumulator="fp32",
            reduction_ops="ambient",
            scope="whole-model",
            description="illegal combination",
        )


def test_cast_weather_tensors_leaves_lat_lon_fp32() -> None:
    batch = make_valid_batch()
    typed = _cast_weather_tensors(batch, torch.float16)

    assert next(iter(typed.surf_vars.values())).dtype == torch.float16
    assert next(iter(typed.static_vars.values())).dtype == torch.float16
    assert next(iter(typed.atmos_vars.values())).dtype == torch.float16
    assert typed.metadata.lat.dtype == torch.float32
    assert typed.metadata.lon.dtype == torch.float32


def test_weight_converted_debug_model_accepts_fp32_batch() -> None:
    """AuroraSmallPretrained, no checkpoint: FP32 batch into converted weights."""
    model = _convert_incoming_batch_to_param_dtype(AuroraSmallPretrained())
    batch = make_valid_batch()
    assert next(iter(batch.surf_vars.values())).dtype == torch.float32

    for dtype in (torch.bfloat16, torch.float16):
        model.to(dtype=dtype)
        assert next(model.parameters()).dtype == dtype
        # Untrained AuroraSmallPretrained emits NaNs; this test is the dtype seam.
        predictions = run_rollout(model, batch, steps=1, check_finite=False)
        assert next(iter(predictions[0].surf_vars.values())).dtype == torch.float32
        assert predictions[0].metadata.lat.dtype == torch.float32


def test_precision_variant_names_match_registry() -> None:
    assert set(PRECISION_VARIANT_NAMES) == set(VARIANTS) - {"fp32-baseline"}
    assert set(CUDA_ONLY_VARIANTS) == {"bf16-amp-full", "fp16-weights-amp"}


def test_apply_variant_precision_cuda_only_refuses_on_cpu() -> None:
    batch = make_valid_batch()
    for name in CUDA_ONLY_VARIANTS:
        model = build_debug_variant(name)
        with pytest.raises(RuntimeError, match="refusing to silently run in FP32"):
            run_rollout(model, batch, steps=1, check_finite=False)
