"""Baseline variant config (no HuggingFace, no weights)."""

from __future__ import annotations

from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.evaluation.variants import VARIANTS

_HYGIENE_FLAGS = (
    "torch.inference_mode",
    "cudnn.benchmark=True",
    "torch.set_float32_matmul_precision('highest')",
    "offload_to_cpu=True",
)


def test_fp32_baseline_is_configure_correctly() -> None:
    variant = VARIANTS["fp32-baseline"]
    assert variant.spec == AURORA_PRETRAINED_SPEC
    for flag in _HYGIENE_FLAGS:
        assert flag in variant.description
