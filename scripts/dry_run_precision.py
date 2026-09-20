"""WP2.1: dry-run the six Stage 3 precision variants on the small debug model.

Does **not** call ``--cpu-dry-run`` and does **not** load ``aurora-finetuned``.
Each registered factory's precision helper is applied to untrained
``AuroraSmallPretrained``, synthetic 32×64, one init, two steps.

CUDA-only AMP rows must refuse rather than silently run FP32. This has no
forecast skill — do not quote RMSE.

Example::

    uv run python scripts/dry_run_precision.py --output-dir outputs/wp2-precision-dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from aurora_inference.evaluation.debug_precision import (
    dry_run_debug_variant,
    lookup_precision_variant,
)
from aurora_inference.evaluation.variants import PRECISION_VARIANT_NAMES
from aurora_inference.logging import configure_run_logging, normalize_run_tag, run_artifact_dir

_LOG = logging.getLogger(__name__)

_OUTPUT_DIR = Path("outputs/wp2-precision-dry-run")
_NO_SKILL_BANNER = (
    "=== NO FORECAST SKILL === WP2.1 uses AuroraSmallPretrained + SyntheticSource. "
    "Run/refuse and DVO only; not a result."
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="artefact root (default: outputs/wp2-precision-dry-run)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="optional run label; writes under <output-dir>/<tag>/",
    )
    return parser.parse_args(argv)


def _summary_payload(results: list[Any]) -> dict[str, Any]:
    return {
        "cpu_dry_run": False,
        "model": "aurora-small-pretrained",
        "n_steps": 2,
        "skill": False,
        "variants": [
            {
                "declared_vs_observed": item.declared_vs_observed,
                "name": item.variant,
                "outcome": item.outcome,
                "reason": item.reason,
                "run_log": str(item.run_log_path),
            }
            for item in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tag = normalize_run_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    output_dir = run_artifact_dir(args.output_dir.expanduser(), tag)
    configure_run_logging(output_dir / "dry_run_precision.log", tag=tag)
    _LOG.warning(_NO_SKILL_BANNER)

    results = []
    for name in PRECISION_VARIANT_NAMES:
        _LOG.info("dry-run %s", name)
        result = dry_run_debug_variant(lookup_precision_variant(name), output_dir)
        results.append(result)
        _LOG.info(
            "%s outcome=%s dvo=%s reason=%s",
            result.variant,
            result.outcome,
            result.declared_vs_observed,
            result.reason,
        )

    summary_path = output_dir / "session.json"
    summary_path.write_text(
        json.dumps(_summary_payload(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _LOG.info("wrote summary %s", summary_path)
    _LOG.warning(_NO_SKILL_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
