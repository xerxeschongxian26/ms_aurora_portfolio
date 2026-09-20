"""Stage 3 precision guards: dtype hooks, declared-versus-observed, inverted zero.

Scoring storage width is FP32 (see ``cast_prediction_to_fp32``). Accumulation
width is FP64 (see ``evaluation.metrics.MSE``). This module does not persist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from aurora import Batch
from torch import nn

from aurora_inference.evaluation.variants import VariantConfig
from aurora_inference.runlog import DeclaredPrecision, ObservedModuleDType

__all__ = [
    "DECLARED_VS_OBSERVED_PASS",
    "DTypeHookSession",
    "DeclaredObservedMismatchError",
    "INVERTED_ZERO_PASS",
    "INVERTED_ZERO_SUSPECT",
    "check_declared_versus_observed",
    "declared_precision",
    "install_dtype_hooks",
    "inverted_zero_verdict",
]

DECLARED_VS_OBSERVED_PASS = "PASS"
INVERTED_ZERO_PASS = "PASS"
INVERTED_ZERO_SUSPECT = "SUSPECT — variant may not have applied"

_BASELINE_VARIANT = "fp32-baseline"
_BOUNDARY_MODULES = ("encoder", "backbone", "decoder")
_REDUCED_NAMES = frozenset({"bf16", "fp16"})
_TORCH_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}


class DeclaredObservedMismatchError(RuntimeError):
    """Variant metadata does not match tensors observed on this run."""

    def __init__(self, message: str, *, declared: object, observed: object) -> None:
        super().__init__(message)
        self.declared = declared
        self.observed = observed


@dataclass
class DTypeHookSession:
    """Forward-hook handle. ``observations`` is filled on the first forward."""

    observations: list[ObservedModuleDType] = field(default_factory=list)
    _hooks: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    _recorded: set[str] = field(default_factory=set)

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def declared_precision(variant: VariantConfig) -> DeclaredPrecision:
    """Copy the five typed fields into the run-log schema."""
    return DeclaredPrecision(
        weights_dtype=variant.weights_dtype,
        matmul_operand=variant.matmul_operand,
        accumulator=variant.accumulator,
        reduction_ops=variant.reduction_ops,
        scope=variant.scope,
    )


def install_dtype_hooks(model: nn.Module) -> DTypeHookSession:
    """Record encoder / backbone / decoder / LayerNorm dtypes, once per run.

    LayerNorm is hooked because AMP pins it to FP32 and weight conversion does not.
    """
    session = DTypeHookSession()
    for name in _BOUNDARY_MODULES:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            session._hooks.append(module.register_forward_hook(_boundary_hook(session, name)))
    for qualified_name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        region = qualified_name.split(".", 1)[0]
        if region not in _BOUNDARY_MODULES:
            continue
        key = f"layernorm:{region}"
        session._hooks.append(
            module.register_forward_hook(_boundary_hook(session, "layernorm", key, qualified_name))
        )
    return session


def check_declared_versus_observed(
    variant: VariantConfig,
    model: nn.Module,
    observations: Sequence[ObservedModuleDType],
) -> str:
    """Assert ``weights_dtype`` and ``scope``. Record-only fields are not asserted.

    ``bf16-weights`` is converted resident weights (``next(parameters()).dtype``).
    microsoft-aurora 1.8.0's remapped ``bf16_mode`` leaves parameters in FP32 and
    must fail this check.

    ``accumulator``, ``reduction_ops``, and TF32 ``matmul_operand`` are not
    observable from Python — callers log the declared value beside ``torch.backends``
    readback and do not assert them.
    """
    observed_weights = _dtype_name(next(model.parameters()).dtype)
    expected_reduced = _expected_reduced_boundaries(variant)
    observed_reduced = _observed_reduced_boundaries(observations)
    mismatches: list[str] = []
    if observed_weights != variant.weights_dtype:
        mismatches.append(
            f"weights_dtype: declared {variant.weights_dtype!r}, observed {observed_weights!r}"
        )
    if observed_reduced != expected_reduced:
        mismatches.append(
            "scope: declared "
            f"{variant.scope!r} expects reduced modules {sorted(expected_reduced)}, "
            f"observed {sorted(observed_reduced)}"
        )
    if mismatches:
        raise DeclaredObservedMismatchError(
            "; ".join(mismatches),
            declared={
                "weights_dtype": variant.weights_dtype,
                "scope": variant.scope,
                "reduced_modules": sorted(expected_reduced),
            },
            observed={
                "weights_dtype": observed_weights,
                "reduced_modules": sorted(observed_reduced),
                "formats": [item.model_dump(mode="json") for item in observations],
            },
        )
    return DECLARED_VS_OBSERVED_PASS


def inverted_zero_verdict(*, variant_name: str, rmses: Sequence[float]) -> str:
    """Exact-zero RMSE vs baseline means a reduced-precision setting never applied.

    ``fp32-baseline`` is exempt: Stage 2's floor is exactly zero. A real numerical
    change lands around 1e-4, never cleanly at zero.
    """
    if variant_name == _BASELINE_VARIANT or not rmses:
        return INVERTED_ZERO_PASS
    if all(float(value) == 0.0 for value in rmses):
        return INVERTED_ZERO_SUSPECT
    return INVERTED_ZERO_PASS


def _boundary_hook(
    session: DTypeHookSession,
    module_label: str,
    record_key: str | None = None,
    qualified_name: str | None = None,
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    key = record_key if record_key is not None else module_label

    def hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        if key in session._recorded:
            return
        in_dtype = _first_tensor_dtype(inputs)
        out_dtype = _first_tensor_dtype(output)
        if in_dtype is None and out_dtype is None:
            return
        session._recorded.add(key)
        session.observations.append(
            ObservedModuleDType(
                module=module_label,
                input_dtype=_dtype_name(in_dtype) if in_dtype is not None else "unknown",
                output_dtype=_dtype_name(out_dtype) if out_dtype is not None else "unknown",
                qualified_name=qualified_name,
            )
        )

    return hook


def _expected_reduced_boundaries(variant: VariantConfig) -> set[str]:
    if variant.scope == "matmul-only":
        return set()
    uses_reduced = (
        variant.weights_dtype in _REDUCED_NAMES or variant.matmul_operand in _REDUCED_NAMES
    )
    if not uses_reduced:
        return set()
    if variant.scope == "backbone-only":
        return {"backbone"}
    if variant.scope == "whole-model":
        return set(_BOUNDARY_MODULES)
    return set()


def _observed_reduced_boundaries(observations: Sequence[ObservedModuleDType]) -> set[str]:
    reduced: set[str] = set()
    for observation in observations:
        if observation.module not in _BOUNDARY_MODULES:
            continue
        if observation.input_dtype in _REDUCED_NAMES or observation.output_dtype in _REDUCED_NAMES:
            reduced.add(observation.module)
    return reduced


def _first_tensor_dtype(value: object) -> torch.dtype | None:
    if isinstance(value, torch.Tensor):
        return value.dtype
    if isinstance(value, Batch):
        tensor = next(iter(value.surf_vars.values()))
        return tensor.dtype if isinstance(tensor, torch.Tensor) else None
    if isinstance(value, tuple | list):
        for item in value:
            dtype = _first_tensor_dtype(item)
            if dtype is not None:
                return dtype
    return None


def _dtype_name(dtype: torch.dtype) -> str:
    return _TORCH_DTYPE_NAMES.get(dtype, str(dtype))
