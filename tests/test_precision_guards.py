"""Stage 3 WP1 precision guards (no GPU, no finetuned weights)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch
from conftest import make_valid_batch
from torch import nn

from aurora_inference.evaluation.grids import batch_to_dataset
from aurora_inference.evaluation.precision import (
    INVERTED_ZERO_SUSPECT,
    DeclaredObservedMismatchError,
    check_declared_versus_observed,
    inverted_zero_verdict,
)
from aurora_inference.evaluation.variants import VARIANTS, _cast_weather_tensors
from aurora_inference.inference.forward import (
    NonFinitePredictionError,
    cast_prediction_to_fp32,
    check_prediction_finite,
    finalize_rollout_predictions,
)
from aurora_inference.runlog import ObservedModuleDType


def test_fake_noop_variant_is_suspect() -> None:
    verdict = inverted_zero_verdict(variant_name="bf16-weights", rmses=[0.0, 0.0, 0.0])
    assert verdict == INVERTED_ZERO_SUSPECT
    assert inverted_zero_verdict(variant_name="fp32-baseline", rmses=[0.0]) == "PASS"
    assert inverted_zero_verdict(variant_name="bf16-weights", rmses=[1e-4]) == "PASS"


def test_injected_infinity_is_caught_and_session_continues() -> None:
    init_time = datetime(2022, 6, 15, 12, 0)
    finite = make_valid_batch()
    exploding = make_valid_batch()
    exploding.surf_vars["msl"][:] = torch.inf

    outcomes: list[str] = []
    caught: NonFinitePredictionError | None = None
    for name, batches in (("ok", [finite]), ("fp16-overflow", [finite, exploding])):
        try:
            for step, pred in enumerate(batches, start=1):
                check_prediction_finite(pred, step=step, init_time=init_time)
            outcomes.append(name)
        except NonFinitePredictionError as exc:
            caught = exc
            outcomes.append("caught")
    assert outcomes == ["ok", "caught"]
    assert caught is not None
    assert caught.variable == "msl"
    assert caught.step == 2
    assert caught.init_time == init_time


def test_declared_bf16_with_fp32_params_fails() -> None:
    model = nn.Linear(2, 2)
    assert next(model.parameters()).dtype == torch.float32
    with pytest.raises(DeclaredObservedMismatchError, match="weights_dtype"):
        check_declared_versus_observed(VARIANTS["bf16-weights"], model, observations=[])


def test_fp16_prediction_is_fp32_at_persist_and_score() -> None:
    native = _cast_weather_tensors(make_valid_batch(), torch.float16)
    assert next(iter(native.surf_vars.values())).dtype == torch.float16

    stored = cast_prediction_to_fp32(native)
    assert next(iter(stored.surf_vars.values())).dtype == torch.float32
    assert next(iter(native.surf_vars.values())).dtype == torch.float16
    assert stored.metadata.lat.dtype == torch.float32

    scored = batch_to_dataset(native)
    assert scored["2t"].dtype == np.float32
    assert scored["msl"].dtype == np.float32


def test_finalize_after_timer_still_guards() -> None:
    init_time = datetime(2022, 6, 15, 12, 0)
    finite = _cast_weather_tensors(make_valid_batch(), torch.float16)
    exploding = _cast_weather_tensors(make_valid_batch(), torch.float16)
    exploding.surf_vars["msl"][:] = torch.inf
    stored = finalize_rollout_predictions([finite], init_time=init_time)
    assert next(iter(stored[0].surf_vars.values())).dtype == torch.float32
    with pytest.raises(NonFinitePredictionError, match="msl"):
        finalize_rollout_predictions([finite, exploding], init_time=init_time)


def test_backbone_only_dvo_counts_reduced_layernorm() -> None:
    """Autocast can leave the backbone wrapper in FP32; inner LayerNorm is the signal."""
    model = nn.Linear(2, 2)
    observations = [
        ObservedModuleDType(module="encoder", input_dtype="fp32", output_dtype="fp32"),
        ObservedModuleDType(module="backbone", input_dtype="fp32", output_dtype="fp32"),
        ObservedModuleDType(
            module="layernorm",
            input_dtype="bf16",
            output_dtype="bf16",
            qualified_name="backbone.encoder_layers.0.blocks.0.norm1.ln",
        ),
        ObservedModuleDType(module="decoder", input_dtype="fp32", output_dtype="fp32"),
    ]
    assert (
        check_declared_versus_observed(VARIANTS["bf16-amp-backbone"], model, observations) == "PASS"
    )


def test_backbone_only_dvo_rejects_reduced_encoder_layernorm() -> None:
    model = nn.Linear(2, 2)
    observations = [
        ObservedModuleDType(
            module="layernorm",
            input_dtype="bf16",
            output_dtype="bf16",
            qualified_name="encoder.surf_norm",
        )
    ]
    with pytest.raises(DeclaredObservedMismatchError, match="scope"):
        check_declared_versus_observed(VARIANTS["bf16-amp-backbone"], model, observations)
