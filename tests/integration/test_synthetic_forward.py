"""Slow CPU integration test for ``scripts/synthetic_forward.py``.

Runs a real pinned-checkpoint load and one ``model(batch)`` step. First run may
download from HuggingFace into ``~/.cache/huggingface/``.
Excluded from default pytest CI via ``addopts = -m 'not slow'``.
Run with ``make test-slow``. This tests the synthetic CLI, not ``run_rollout``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "synthetic_forward.py"
_EXPECTED_GRID = (32, 64)


def _load_synthetic_forward_module() -> ModuleType:
    """Load the CLI script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("synthetic_forward", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"cannot load synthetic_forward script from {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_synthetic_forward_runs_one_step_with_t1_output() -> None:
    """Plumbing proof: real load + one CPU forward; output has T=1 and the input grid."""
    synthetic_forward = _load_synthetic_forward_module()
    result = synthetic_forward.run_synthetic_forward()

    assert result.input_surf_shape[1] == 2  # previous time step and current time step
    assert result.input_atmos_shape[1] == 2
    assert result.output_surf_shape[1] == 1
    assert result.output_atmos_shape[1] == 1
    assert result.input_surf_shape[-2:] == _EXPECTED_GRID
    assert result.output_surf_shape[-2:] == _EXPECTED_GRID
    assert result.output_atmos_shape[-2:] == _EXPECTED_GRID
    assert result.load_seconds >= 0.0
    assert result.forward_seconds >= 0.0
    assert result.peak_rss_bytes > 0
