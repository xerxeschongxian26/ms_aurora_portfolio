"""Process-level telemetry for inference runs."""

from __future__ import annotations

import logging
import platform
import resource
import sys
from pathlib import Path

__all__ = [
    "configure_run_logging",
    "normalize_run_tag",
    "peak_rss_bytes",
    "run_artifact_dir",
]


def normalize_run_tag(tag: str | None) -> str | None:
    """Return a stripped tag, or ``None`` if empty. Reject path separators."""
    if tag is None:
        return None
    stripped = tag.strip()
    if stripped == "":
        return None
    if "/" in stripped or "\\" in stripped or ".." in stripped:
        msg = f"run tag must not contain path separators or '..' (received {tag!r})"
        raise ValueError(msg)
    return stripped


def run_artifact_dir(base: Path, tag: str | None) -> Path:
    """Return ``base`` or ``base / tag``, creating the directory."""
    directory = base / tag if tag is not None else base
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_run_logging(log_path: Path, *, tag: str | None = None) -> None:
    """Send INFO logs to stdout and ``log_path``. Emit ``tag`` if set."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=(
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ),
        force=True,
    )
    if tag is not None:
        logging.getLogger(__name__).info("run tag: %s", tag)


def peak_rss_bytes() -> int:
    """Return peak resident set size (RSS) in bytes (platform-normalized).

    ``ru_maxrss`` units differ by OS: macOS reports bytes; Linux reports kilobytes.
    This helper normalises the value to bytes.

    Notes:
        - Statistics are per-process. External apps (e.g. a RAM-heavy browser) do not
          affect the recorded peak, though heavy system load may slow wall-clock
          timings.
        - Peak RSS is a high-water mark since this Python process started (imports,
          model load, forward), not current memory at call time. It never decreases.
    """
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if platform.system() == "Darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024
