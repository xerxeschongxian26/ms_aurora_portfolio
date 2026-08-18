"""Tests for ``_validate_static_vars`` (WP2).

Exercises ``_validate_static_vars`` directly with synthetic fixtures: this private
helper carries the actual contract (exact keys, shape, dtype, finiteness) that
``static_vars.py`` owns independently of any caller. ``get_hres_t0_static``'s own
HF fetch is a real network/disk call and is deliberately out of scope for this
unit-test file — see the WP2 closeout note in stage1_forecastpipeline.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora_inference.data.static_vars import (
    EXPECTED_DIMENSIONS,
    EXPECTED_DTYPE,
    EXPECTED_KEYS,
    InvalidStaticVarsError,
    _validate_static_vars,
)


def _valid_static_vars() -> dict[str, np.ndarray]:
    """A dict containing synthetic arrays satisfying the contract"""
    return {key: np.zeros(EXPECTED_DIMENSIONS, dtype=EXPECTED_DTYPE) for key in EXPECTED_KEYS}


def test_validate_static_vars_accepts_contractually_valid_input() -> None:
    _validate_static_vars(_valid_static_vars())  # must not raise


def test_validate_static_vars_rejects_missing_key() -> None:
    """static_vars is missing a key required in EXPECTED_KEYS"""
    static_vars = _valid_static_vars()
    del static_vars[EXPECTED_KEYS[0]]

    with pytest.raises(InvalidStaticVarsError, match=r"key mismatch"):
        _validate_static_vars(static_vars)


def test_validate_static_vars_rejects_unexpected_extra_key() -> None:
    """static_vars has an extra key, not required in EXPECTED_KEYS"""
    static_vars = _valid_static_vars()
    static_vars["extra_key"] = np.zeros(EXPECTED_DIMENSIONS, dtype=EXPECTED_DTYPE)

    with pytest.raises(InvalidStaticVarsError, match=r"key mismatch"):
        _validate_static_vars(static_vars)


def test_validate_static_vars_rejects_wrong_shape() -> None:
    "Transpose one value array, function should throw an error"
    static_vars = _valid_static_vars()
    key = EXPECTED_KEYS[0]
    static_vars[key] = static_vars[key].T  # transpose -> (1440, 721), wrong shape

    with pytest.raises(InvalidStaticVarsError, match=rf"static_vars\['{key}'\]: shape"):
        _validate_static_vars(static_vars)


def test_validate_static_vars_rejects_wrong_dtype() -> None:
    "Cast one value to incorrect dtype, function should throw an error"
    static_vars = _valid_static_vars()
    key = EXPECTED_KEYS[0]
    static_vars[key] = static_vars[key].astype(np.float64)

    with pytest.raises(InvalidStaticVarsError, match=rf"static_vars\['{key}'\]: dtype"):
        _validate_static_vars(static_vars)


def test_validate_static_vars_rejects_non_finite_value() -> None:
    """One non-finite case (NaN) suffices: NaN/+inf/-inf all fail the same np.isfinite() check."""
    static_vars = _valid_static_vars()
    key = EXPECTED_KEYS[0]
    static_vars[key][0, 0] = np.nan

    with pytest.raises(InvalidStaticVarsError, match=rf"static_vars\['{key}'\]: values"):
        _validate_static_vars(static_vars)
