"""Model-factory registry for Stage 2/3 inference variants.

Adding a variant is a dictionary insertion. The Q2 harness looks up a name
and calls ``model_factory`` without modification. Schedule (which inits)
stays in rollout TOML — not here.

The five typed fields are the mechanism story. ``description`` is prose only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import torch
from aurora import Aurora, Batch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec
from aurora_inference.model.loader import load_model

__all__ = [
    "VARIANTS",
    "VariantConfig",
    "Accumulator",
    "MatmulOperand",
    "PrecisionScope",
    "ReductionOps",
    "WeightsDType",
]

WeightsDType = Literal["fp32", "bf16", "fp16"]
MatmulOperand = Literal["fp32", "tf32", "bf16", "fp16"]
Accumulator = Literal["fp32", "unguaranteed"]
ReductionOps = Literal["fp32-pinned", "ambient", "fp32-pinned-partial"]
PrecisionScope = Literal["whole-model", "backbone-only", "matmul-only"]

_WEIGHTS_DTYPES = frozenset(get_args(WeightsDType))
_MATMUL_OPERANDS = frozenset(get_args(MatmulOperand))
_ACCUMULATORS = frozenset(get_args(Accumulator))
_REDUCTION_OPS = frozenset(get_args(ReductionOps))
_SCOPES = frozenset(get_args(PrecisionScope))
_WEIGHT_CONVERTED = frozenset({"bf16", "fp16"})
_WEIGHT_CONVERTED_REDUCTIONS = frozenset({"ambient", "fp32-pinned-partial"})


@dataclass(frozen=True)
class VariantConfig:
    """One named way to construct an Aurora module for a fidelity run."""

    name: str
    model_factory: Callable[[], Aurora]
    spec: ModelSpec
    weights_dtype: WeightsDType
    matmul_operand: MatmulOperand
    accumulator: Accumulator
    reduction_ops: ReductionOps
    scope: PrecisionScope
    description: str

    def __post_init__(self) -> None:
        _require_allowed(self.name, "weights_dtype", self.weights_dtype, _WEIGHTS_DTYPES)
        _require_allowed(self.name, "matmul_operand", self.matmul_operand, _MATMUL_OPERANDS)
        _require_allowed(self.name, "accumulator", self.accumulator, _ACCUMULATORS)
        _require_allowed(self.name, "reduction_ops", self.reduction_ops, _REDUCTION_OPS)
        _require_allowed(self.name, "scope", self.scope, _SCOPES)
        if self.weights_dtype in _WEIGHT_CONVERTED:
            if self.accumulator != "unguaranteed":
                msg = (
                    f"{self.name}: weight-converted variants must set "
                    f"accumulator='unguaranteed' (received {self.accumulator!r})"
                )
                raise ValueError(msg)
            if self.reduction_ops not in _WEIGHT_CONVERTED_REDUCTIONS:
                msg = (
                    f"{self.name}: weight-converted variants must set reduction_ops to "
                    f"'ambient' or 'fp32-pinned-partial' (received {self.reduction_ops!r})"
                )
                raise ValueError(msg)


def _require_allowed(name: str, field: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        msg = f"{name}: {field} must be one of {{{allowed_str}}} (received {value!r})"
        raise ValueError(msg)


def _cast_weather_tensors(batch: Batch, dtype: object) -> Batch:
    """Cast surf/static/atmos tensors; leave ``metadata.lat`` / ``lon`` unchanged.

    Aurora's ``Batch.type`` uses ``_fmap``, which also casts coordinates. The
    encoder then asserts lat/lon are float32 or float64 (``Latitude num.
    unstable``). Weight-converted variants need weather arrays at the parameter
    width without touching those coordinates.
    """
    return Batch(
        surf_vars={key: tensor.type(dtype) for key, tensor in batch.surf_vars.items()},
        static_vars={key: tensor.type(dtype) for key, tensor in batch.static_vars.items()},
        atmos_vars={key: tensor.type(dtype) for key, tensor in batch.atmos_vars.items()},
        metadata=batch.metadata,
    )


def _convert_incoming_batch_to_param_dtype(model: Aurora) -> Aurora:
    """Make rollout/forward ``batch.type(param.dtype)`` keep lat/lon in float32.

    ``aurora.rollout`` and ``Aurora.forward`` both call ``batch_transform_hook``
    then ``batch.type``. The hook returns a batch whose ``type`` casts weather
    only, so the harness stays unchanged.
    """
    inner_hook = model.batch_transform_hook

    def hook(batch: Batch) -> Batch:
        converted = inner_hook(batch)
        wrapped = Batch(
            surf_vars=converted.surf_vars,
            static_vars=converted.static_vars,
            atmos_vars=converted.atmos_vars,
            metadata=converted.metadata,
        )

        def type_weather_only(dtype: object) -> Batch:
            return _cast_weather_tensors(wrapped, dtype)

        wrapped.type = type_weather_only
        return wrapped

    model.batch_transform_hook = hook
    return model


def _wrap_forward_in_autocast(model: Aurora, *, dtype: torch.dtype) -> Aurora:
    """Wrap ``model.forward`` in CUDA autocast. Refuse on CPU rather than run FP32."""
    inner_forward = model.forward

    def forward(batch: Batch) -> Batch:
        if not torch.cuda.is_available():
            msg = "whole-forward autocast requires CUDA; refusing to silently run in FP32"
            raise RuntimeError(msg)
        with torch.autocast("cuda", dtype=dtype):
            return inner_forward(batch)

    model.forward = forward
    return model


def _fp32_baseline_factory() -> Aurora:
    torch.set_float32_matmul_precision("highest")
    return load_model("aurora-finetuned")


def _tf32_matmul_factory() -> Aurora:
    torch.set_float32_matmul_precision("high")
    return load_model("aurora-finetuned")


def _bf16_amp_backbone_factory() -> Aurora:
    model = load_model("aurora-finetuned")
    model.autocast = True
    return model


def _bf16_amp_full_factory() -> Aurora:
    model = load_model("aurora-finetuned")
    return _wrap_forward_in_autocast(model, dtype=torch.bfloat16)


def _bf16_weights_factory() -> Aurora:
    # microsoft-aurora==1.8.0 remaps Aurora(bf16_mode=True) to backbone autocast.
    # Issue #127 is about converting resident weights; do that in place.
    model = load_model("aurora-finetuned")
    model = model.to(dtype=torch.bfloat16)
    return _convert_incoming_batch_to_param_dtype(model)


def _fp16_weights_factory() -> Aurora:
    model = load_model("aurora-finetuned")
    model.half()
    return _convert_incoming_batch_to_param_dtype(model)


def _fp16_weights_amp_factory() -> Aurora:
    model = load_model("aurora-finetuned")
    model.half()
    model = _convert_incoming_batch_to_param_dtype(model)
    return _wrap_forward_in_autocast(model, dtype=torch.float16)


VARIANTS: dict[str, VariantConfig] = {
    "fp32-baseline": VariantConfig(
        name="fp32-baseline",
        model_factory=_fp32_baseline_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="fp32",
        matmul_operand="fp32",
        accumulator="fp32",
        reduction_ops="fp32-pinned",
        scope="whole-model",
        description=(
            "Full-precision Aurora 0.25 deg FT — THE BASELINE. Operational flags "
            "not captured by the typed fields: torch.inference_mode; "
            "cudnn.benchmark=True; offload_to_cpu=True in run_rollout. "
            "float32_matmul_precision='highest' is this row's matmul_operand=fp32, "
            "not a flag to copy onto other rows."
        ),
    ),
    "tf32-matmul": VariantConfig(
        name="tf32-matmul",
        model_factory=_tf32_matmul_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="fp32",
        matmul_operand="tf32",
        accumulator="fp32",
        reduction_ops="fp32-pinned",
        scope="matmul-only",
        description=(
            "Allow TF32 in matrix multiplications via "
            "torch.set_float32_matmul_precision('high'). Weights stay FP32."
        ),
    ),
    "bf16-amp-backbone": VariantConfig(
        name="bf16-amp-backbone",
        model_factory=_bf16_amp_backbone_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="fp32",
        matmul_operand="bf16",
        accumulator="fp32",
        reduction_ops="fp32-pinned",
        scope="backbone-only",
        description=(
            "Aurora(autocast=True) after loading. BF16 applies to the backbone only; "
            "encoder and decoder stay FP32. Downloaded weights stay FP32."
        ),
    ),
    "bf16-amp-full": VariantConfig(
        name="bf16-amp-full",
        model_factory=_bf16_amp_full_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="fp32",
        matmul_operand="bf16",
        accumulator="fp32",
        reduction_ops="fp32-pinned",
        scope="whole-model",
        description=(
            "Wrap the whole forward pass in "
            "torch.autocast('cuda', dtype=torch.bfloat16). Downloaded weights stay FP32."
        ),
    ),
    "bf16-weights": VariantConfig(
        name="bf16-weights",
        model_factory=_bf16_weights_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="bf16",
        matmul_operand="bf16",
        accumulator="unguaranteed",
        reduction_ops="ambient",
        scope="whole-model",
        description=(
            "Resident weights converted to BF16 in place (original bf16_mode=True "
            "meaning; microsoft-aurora 1.8.0 remaps that constructor flag to "
            "backbone autocast). Input weather tensors are cast to match; lat/lon "
            "stay float32."
        ),
    ),
    "fp16-weights": VariantConfig(
        name="fp16-weights",
        model_factory=_fp16_weights_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="fp16",
        matmul_operand="fp16",
        accumulator="unguaranteed",
        reduction_ops="ambient",
        scope="whole-model",
        description=(
            "Resident weights converted with model.half(). Input weather tensors "
            "are cast to match; lat/lon stay float32."
        ),
    ),
    "fp16-weights-amp": VariantConfig(
        name="fp16-weights-amp",
        model_factory=_fp16_weights_amp_factory,
        spec=AURORA_PRETRAINED_SPEC,
        weights_dtype="fp16",
        matmul_operand="fp16",
        accumulator="unguaranteed",
        reduction_ops="fp32-pinned-partial",
        scope="whole-model",
        description=(
            "model.half(), then wrap the forward pass in automatic mixed precision. "
            "This is the starter-kit snippet; AMP may lift some ops back to FP32 "
            "while resident weights stay FP16."
        ),
    ),
}
