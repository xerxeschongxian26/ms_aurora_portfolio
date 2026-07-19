"""Slow CPU integration test for ``scripts/toy_forward.py`` (WP5).

Runs a real pinned-checkpoint load and one forward pass. First run may download
from HuggingFace into ``~/.cache/huggingface/``. Excluded from default pytest /
CI via ``addopts = -m 'not slow'``; run with ``make test-slow``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "toy_forward.py"
_EXPECTED_GRID = (32, 64)


def _load_toy_forward_module() -> ModuleType:
    """Load the CLI script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("toy_forward", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"cannot load toy_forward script from {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_toy_forward_cpu_runs_and_output_has_time_dim_1() -> None:
    """Plumbing proof: real load + forward on CPU; output time dim is 1."""
    toy_forward = _load_toy_forward_module()
    result = toy_forward.run_toy_forward()

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
