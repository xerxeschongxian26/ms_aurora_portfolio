"""Tests for process-level telemetry helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aurora_inference.logging import (
    configure_run_logging,
    normalize_run_tag,
    peak_rss_bytes,
    run_artifact_dir,
)


def test_peak_rss_bytes_is_positive() -> None:
    assert peak_rss_bytes() > 0


def test_normalize_run_tag_strips_and_rejects_paths() -> None:
    assert normalize_run_tag(None) is None
    assert normalize_run_tag("  ") is None
    assert normalize_run_tag(" b1-s4 ") == "b1-s4"
    with pytest.raises(ValueError, match="path separators"):
        normalize_run_tag("b1/s4")


def test_run_artifact_dir_uses_tag_subdirectory(tmp_path: Path) -> None:
    assert run_artifact_dir(tmp_path, None) == tmp_path
    tagged = run_artifact_dir(tmp_path, "b1-s4")
    assert tagged == tmp_path / "b1-s4"
    assert tagged.is_dir()


def test_configure_run_logging_writes_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_run_logging(log_path, tag="b1-s4")
    logging.getLogger("test_logging").info("hello from test")
    text = log_path.read_text(encoding="utf-8")
    assert "run tag: b1-s4" in text
    assert "hello from test" in text
