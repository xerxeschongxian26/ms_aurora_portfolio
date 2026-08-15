"""WP5 quick-look map: one HRES-T0 Batch → one global PNG.

Throwaway plausibility gate, not a skill plot. Loads a single Batch via
``HresT0Source`` (caller injects GCS zarr + HF static) and renders ``2t``
(2 m temperature) with a coastline overlay — the same global-field check as
the earthkit scratch notebook, using matplotlib + cartopy.

Requires the ``viz`` extra (and ``dev`` for zarr/gcsfs until those move to
core)::

    uv sync --extra dev --extra viz
    python scripts/plot_forecast.py

Not imported by ``src/``. Core runtime must not grow a matplotlib/cartopy
dependency from this script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - local extra, not CI
    msg = "plot_forecast.py requires the viz extra: uv sync --extra viz"
    raise SystemExit(msg) from exc
from aurora import Batch

from aurora_inference.config import GCS_STORE_LINK
from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.data.hres_t0 import HresT0Source, open_connection_to_gcs
from aurora_inference.data.static_vars import get_hres_t0_static

_LOG = logging.getLogger(__name__)

_PLOT_VARIABLE = "2t"
_KELVIN_OFFSET = 273.15
_DEFAULT_INIT_TIME = datetime(2022, 6, 15, 12, 0)
_DEFAULT_OUTPUT = Path("outputs/hres_t0_2t.png")


def load_hres_t0_batch(init_time: datetime) -> Batch:
    """Build one contract Batch from live HRES-T0 (GCS zarr + HF static)."""
    _LOG.info("opening HRES-T0 zarr at %s", GCS_STORE_LINK)
    zarr_data = open_connection_to_gcs(GCS_STORE_LINK)
    _LOG.info("loading official aurora-0.25 static vars")
    static_vars = get_hres_t0_static()
    source = HresT0Source(ZARR_DATA=zarr_data, STATIC_VARS=static_vars)
    _LOG.info("HresT0Source.load(init_time=%s)", init_time.isoformat())
    return source.load(init_time, AURORA_PRETRAINED_SPEC)


def plot_batch_2t(batch: Batch, output_path: Path) -> None:
    """Render global 2 m temperature (°C) at the last time index (t1 / T=1)."""
    if _PLOT_VARIABLE not in batch.surf_vars:
        keys = sorted(batch.surf_vars)
        msg = f"batch is missing surf_vars[{_PLOT_VARIABLE!r}]; have {keys}"
        raise KeyError(msg)

    lat = batch.metadata.lat.detach().cpu()
    lon = batch.metadata.lon.detach().cpu()
    field_celsius = batch.surf_vars[_PLOT_VARIABLE][0, -1].detach().cpu() - _KELVIN_OFFSET

    # Plot copy only: HRES-T0 lon is [0, 360); PlateCarree is Greenwich-centered.
    lon_plot = lon.where(lon <= 180, lon - 360)
    order = lon_plot.argsort()
    lon_sorted = lon_plot[order].numpy()
    field_sorted = field_celsius[:, order].numpy()
    lat_np = lat.numpy()

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    mesh = ax.pcolormesh(
        lon_sorted,
        lat_np,
        field_sorted,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        shading="auto",
    )
    ax.coastlines()
    ax.set_global()
    fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, label="2m temperature (°C)")
    valid_time = batch.metadata.time[0]
    ax.set_title(f"HRES-T0 {_PLOT_VARIABLE} at {valid_time.isoformat()} (last time index)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("wrote %s", output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-time",
        type=datetime.fromisoformat,
        default=_DEFAULT_INIT_TIME,
        help="HRES-T0 t1 (00 or 12 UTC, 2022). Default: 2022-06-15T12:00:00",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"PNG path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    batch = load_hres_t0_batch(args.init_time)
    plot_batch_2t(batch, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
