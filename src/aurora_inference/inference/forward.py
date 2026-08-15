"""Autoregressive forecast wrapper around Aurora's ``rollout`` helper.

Each step consumes two input timesteps (T=2) and yields one predicted step (T=1)
at a 6-hour lead. ``validate_batch`` gates the *input* at this boundary; output
batches have T=1 and therefore cannot pass the input contract.
"""

from __future__ import annotations

import torch
from aurora import Aurora, Batch, rollout

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec, validate_batch

__all__ = ["run_forecast"]


def run_forecast(
    model: Aurora,
    batch: Batch,
    steps: int,
    *,
    spec: ModelSpec = AURORA_PRETRAINED_SPEC,
) -> list[Batch]:
    """Roll the model forward ``steps`` times and return one ``Batch`` per lead time.

    Wraps :func:`aurora.rollout.rollout` under ``torch.inference_mode()``. Device
    placement follows the model (CPU for WP4/WP5, GPU in WP6).

    Args:
        model: An Aurora checkpoint already in eval mode on the target device
            (typically from :func:`~aurora_inference.model.loader.load_model`).
        batch: Input batch with T=2, validated against ``spec`` before rollout.
        steps: Number of 6-hour lead times to produce (must be >= 1).
        spec: Variable/level contract for the input batch.

    Returns:
        ``steps`` batches, each with time dim 1, in lead-time order
        (6 h, 12 h, …).

    Raises:
        ValueError: ``steps`` is less than 1.
        BatchContractError: ``batch`` fails :func:`validate_batch`.
        RuntimeError: An output batch does not have time dim 1.
    """
    if steps < 1:
        msg = f"steps must be >= 1 (received {steps})"
        raise ValueError(msg)

    validate_batch(batch, spec)

    with torch.inference_mode():
        forecasts = list(rollout(model, batch, steps))

    for pred in forecasts:
        _assert_output_time_dim_is_one(pred)
    return forecasts


def _assert_output_time_dim_is_one(prediction: Batch) -> None:
    """Aurora's single rollout step returns T==1, not the input T==2."""
    for name, tensor in prediction.surf_vars.items():
        if tensor.shape[1] != 1:
            msg = f"output surf_vars[{name!r}] time dim must be 1, got shape {tuple(tensor.shape)}"
            raise RuntimeError(msg)
    for name, tensor in prediction.atmos_vars.items():
        if tensor.shape[1] != 1:
            msg = f"output atmos_vars[{name!r}] time dim must be 1, got shape {tuple(tensor.shape)}"
            raise RuntimeError(msg)
