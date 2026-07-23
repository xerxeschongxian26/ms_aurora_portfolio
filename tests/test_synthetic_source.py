"""Tests for SyntheticSource (WP3 / D4)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest
import torch

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, validate_batch
from aurora_inference.data.base import BatchSource
from aurora_inference.data.synthetic import SyntheticSource

_INIT_TIME = datetime(2020, 6, 15, 12, 0)
_SPEC = AURORA_PRETRAINED_SPEC


def test_synthetic_source_is_batch_source() -> None:
    assert isinstance(SyntheticSource(), BatchSource)


def test_synthetic_source_load_passes_validate_batch() -> None:
    batch = SyntheticSource().load(_INIT_TIME, _SPEC)
    validate_batch(batch, _SPEC)


def test_synthetic_source_load_honors_constructor_grid_keys_and_init_time() -> None:
    """Cover source-specific guarantees that ``validate_batch`` does not pin.

    ``validate_batch`` allows extra keys, any consistent (H, W, B), and only
    checks ``len(metadata.time) == B`` — not the datetime values.
    """
    height, width, batch_size = 32, 64, 1
    source = SyntheticSource(height=height, width=width, batch_size=batch_size)
    batch = source.load(_INIT_TIME, _SPEC)

    assert set(batch.surf_vars) == set(_SPEC.surf_vars)
    assert set(batch.static_vars) == set(_SPEC.static_vars)
    assert set(batch.atmos_vars) == set(_SPEC.atmos_vars)

    reference = batch.surf_vars[_SPEC.surf_vars[0]]
    assert reference.shape[0] == batch_size
    assert reference.shape[2] == height
    assert reference.shape[3] == width
    assert len(batch.metadata.lat) == height
    assert len(batch.metadata.lon) == width

    assert batch.metadata.time == (_INIT_TIME,) * batch_size


def test_synthetic_source_same_seed_is_deterministic() -> None:
    a = SyntheticSource(seed=7).load(_INIT_TIME, _SPEC)
    b = SyntheticSource(seed=7).load(_INIT_TIME, _SPEC)

    for key in _SPEC.surf_vars:
        assert torch.equal(a.surf_vars[key], b.surf_vars[key])
    for key in _SPEC.static_vars:
        assert torch.equal(a.static_vars[key], b.static_vars[key])
    for key in _SPEC.atmos_vars:
        assert torch.equal(a.atmos_vars[key], b.atmos_vars[key])


def test_synthetic_source_different_seed_differs() -> None:
    a = SyntheticSource(seed=7).load(_INIT_TIME, _SPEC)
    b = SyntheticSource(seed=8).load(_INIT_TIME, _SPEC)

    differs = (
        any(not torch.equal(a.surf_vars[key], b.surf_vars[key]) for key in _SPEC.surf_vars)
        or any(not torch.equal(a.static_vars[key], b.static_vars[key]) for key in _SPEC.static_vars)
        or any(not torch.equal(a.atmos_vars[key], b.atmos_vars[key]) for key in _SPEC.atmos_vars)
    )
    assert differs


def test_synthetic_source_load_raises_on_missing_range_entry() -> None:
    """Unknown spec keys must fail at lookup, even for synthetic source"""
    spec = replace(AURORA_PRETRAINED_SPEC, surf_vars=("2t", "unknown_var"))

    with pytest.raises(KeyError):
        SyntheticSource().load(_INIT_TIME, spec)
