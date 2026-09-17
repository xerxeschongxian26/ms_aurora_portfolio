"""Model-factory registry for Stage 2/3 inference variants.

Stage 2 ships ``fp32-baseline`` only. Stage 3 adds rows to ``VARIANTS``; the
Q2 harness looks up a name and calls ``model_factory`` without modification.
Schedule (which inits) stays in rollout TOML — not here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aurora import Aurora

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec
from aurora_inference.model.loader import load_model

__all__ = ["VARIANTS", "VariantConfig"]


@dataclass(frozen=True)
class VariantConfig:
    """One named way to construct an Aurora module for a fidelity run."""

    name: str
    model_factory: Callable[[], Aurora]
    spec: ModelSpec
    description: str


def _fp32_baseline_factory() -> Aurora:
    return load_model("aurora-finetuned")


VARIANTS: dict[str, VariantConfig] = {
    "fp32-baseline": VariantConfig(
        name="fp32-baseline",
        model_factory=_fp32_baseline_factory,
        spec=AURORA_PRETRAINED_SPEC,
        description=(
            "Full-precision Aurora 0.25 deg FT — THE BASELINE. The hygiene flags below "
            "are part of the baseline definition, not optimizations: "
            "torch.inference_mode; cudnn.benchmark=True; "
            "torch.set_float32_matmul_precision('highest'); "
            "offload_to_cpu=True in run_rollout."
        ),
    ),
    # Stage 3 adds rows here (tf32, bf16-autocast, ...):
    # "tf32": VariantConfig(...),
}
