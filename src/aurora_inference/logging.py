"""Process-level telemetry for inference runs."""

from __future__ import annotations

import platform
import resource

__all__ = ["peak_rss_bytes"]


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
