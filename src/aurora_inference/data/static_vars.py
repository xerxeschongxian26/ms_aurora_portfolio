# Load the static variables from HF cache OR download if cache is unavailable


import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
from huggingface_hub import hf_hub_download

from aurora_inference.config import AURORA_HF_REPO_ID, AURORA_HF_REVISION

__all__ = [
    "InvalidStaticVarsError",
    "get_hres_t0_static",
]

EXPECTED_DIMENSIONS = (721, 1440)
EXPECTED_DTYPE = np.dtype("float32")
EXPECTED_KEYS = ("lsm", "slt", "z")


class InvalidStaticVarsError(Exception):
    """Raised when the unpickled static variables violate the expected contract."""


def get_hres_t0_static(cache_dir: Path | None = None) -> dict[str, Any]:
    """Download and load aurora-0.25-static.pickle from Hugging Face.

    Returns a dict of static variable name -> numpy array (e.g. 'z', 'slt', 'lsm').
    Cached under ~/.cache/huggingface/ by default.

    Raises:
        InvalidStaticVarsError: If the unpickled object's keys, shapes, dtypes, or
            values don't match the expected Aurora static-variable contract.
    """
    STATIC_PICKLE_FILENAME = "aurora-0.25-static.pickle"

    path = hf_hub_download(
        repo_id=AURORA_HF_REPO_ID,
        filename=STATIC_PICKLE_FILENAME,
        revision=AURORA_HF_REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    with open(path, "rb") as f:
        static_vars = cast(dict[str, Any], pickle.load(f))

    _validate_static_vars(static_vars)
    return static_vars


def _validate_static_vars(static_vars: dict[str, Any]) -> None:
    """Validate keys, shape, dtype, and finiteness of the unpickled static variables.

    Match-not-superset on keys: missing *and* unexpected keys are both rejected, so a
    format change upstream (e.g. Microsoft adding/renaming a field) fails loudly here
    instead of silently mis-feeding the model.

    Raises:
        InvalidStaticVarsError: With an actionable message naming the offending field.
    """
    received_keys = set(static_vars)
    expected_keys = set(EXPECTED_KEYS)
    if received_keys != expected_keys:
        missing = sorted(expected_keys - received_keys)
        unexpected = sorted(received_keys - expected_keys)
        msg = (
            f"static_vars: key mismatch (missing {missing}, unexpected {unexpected}, "
            f"expected {sorted(expected_keys)})"
        )
        raise InvalidStaticVarsError(msg)

    for key in EXPECTED_KEYS:
        array = static_vars[key]

        if tuple(array.shape) != EXPECTED_DIMENSIONS:
            msg = (
                f"static_vars['{key}']: shape "
                f"(received {tuple(array.shape)}, expected {EXPECTED_DIMENSIONS})"
            )
            raise InvalidStaticVarsError(msg)

        if array.dtype != EXPECTED_DTYPE:
            msg = f"static_vars['{key}']: dtype (received {array.dtype}, expected {EXPECTED_DTYPE})"
            raise InvalidStaticVarsError(msg)

        if not np.isfinite(array).all():
            msg = f"static_vars['{key}']: values (received non-finite, expected finite)"
            raise InvalidStaticVarsError(msg)
