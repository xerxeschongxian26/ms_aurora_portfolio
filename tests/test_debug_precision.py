"""WP2.1 laptop dry-run: each precision variant runs or refuses on the debug model."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_inference.evaluation.debug_precision import (
    CUDA_ONLY_REFUSE_MESSAGE,
    dry_run_debug_variant,
    lookup_precision_variant,
)
from aurora_inference.evaluation.variants import CUDA_ONLY_VARIANTS, PRECISION_VARIANT_NAMES
from aurora_inference.runlog import load_run_log


def test_debug_dry_run_does_not_accept_baseline() -> None:
    with pytest.raises(KeyError, match="fp32-baseline"):
        lookup_precision_variant("fp32-baseline")


def test_debug_dry_run_all_precision_variants_run_or_refuse(tmp_path: Path) -> None:
    """Synthetic 32×64, one init, two steps. No aurora-finetuned weights."""
    for name in PRECISION_VARIANT_NAMES:
        result = dry_run_debug_variant(lookup_precision_variant(name), tmp_path)
        log = load_run_log(result.run_log_path)
        assert log.run.variant == name
        assert log.run.model_name == "aurora-small-pretrained"
        assert log.run.cpu_dry_run is False
        assert log.declared_precision is not None
        assert log.declared_vs_observed is not None
        if name in CUDA_ONLY_VARIANTS:
            assert result.outcome == "refused"
            assert result.reason is not None
            assert CUDA_ONLY_REFUSE_MESSAGE in result.reason
            assert log.declared_vs_observed.startswith("REFUSED")
        else:
            assert result.outcome == "ran"
            assert result.reason is None
            assert log.declared_vs_observed == "PASS"
            assert log.observed_formats
