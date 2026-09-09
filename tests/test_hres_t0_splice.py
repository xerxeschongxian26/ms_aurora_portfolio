"""Offline helpers for HRES-T0 splice progress / byte accounting (no GCS)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import HRES_T0_FIXTURE_ZARR

from aurora_inference.data.hres_t0 import (
    _ATMOS_NAME_MAP,
    _SURF_NAME_MAP,
    open_local_zarr,
)
from aurora_inference.data.hres_t0_splice import (
    _decoded_slab_nbytes,
    _expected_decoded_bytes,
    _format_data_size,
)
from aurora_inference.logging import format_elapsed

_REPO = Path(__file__).resolve().parents[1]
_SPLICE_CLI = _REPO / "scripts" / "toy_hres_t0_splice.py"


def test_format_elapsed_and_data_size() -> None:
    assert format_elapsed(12) == "12s"
    assert format_elapsed(75) == "1m 15s"
    assert _format_data_size(512) == "512 B"
    assert _format_data_size(2048) == "2.0 KiB"
    assert _format_data_size(5 * 1024 * 1024) == "5.0 MiB"
    assert _format_data_size(3 * 1024 * 1024 * 1024) == "3.00 GiB"


def test_expected_decoded_bytes_matches_fixture_shapes() -> None:
    group = open_local_zarr(HRES_T0_FIXTURE_ZARR)
    names = tuple(_SURF_NAME_MAP.values()) + tuple(_ATMOS_NAME_MAP.values())
    n_times = 2
    expected = _expected_decoded_bytes(group, names, n_times)
    manual = 0
    for name in names:
        manual += n_times * _decoded_slab_nbytes(cast(Any, group[name]))
    assert expected == manual
    assert expected > 0


def _load_splice_cli() -> Any:
    spec = importlib.util.spec_from_file_location("toy_hres_t0_splice_cli", _SPLICE_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_toy_hres_t0_splice_writes_progress_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cli = _load_splice_cli()
    monkeypatch.setattr(cli, "_download_splice", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_check_rollout_coverage", lambda _: None)
    assert cli.main(["--tag", "spread-splice"]) == 0
    log_path = tmp_path / "outputs" / "spread-splice" / "toy_hres_t0_splice.log"
    text = log_path.read_text(encoding="utf-8")
    assert "run tag: spread-splice" in text
    assert "Splice HRES-T0; log file:" in text
