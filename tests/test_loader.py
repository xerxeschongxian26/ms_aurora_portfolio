"""Tests for pinned checkpoint config and load_model (WP4).

Default suite never downloads from HuggingFace — load_checkpoint is mocked.
"""

from __future__ import annotations

import string
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from aurora_inference.config import (
    AURORA_HF_REPO_ID,
    AURORA_HF_REVISION,
    CHECKPOINT_REGISTRY,
    ModelName,
    resolve_checkpoint,
    spec_for_model,
)
from aurora_inference.contract import AURORA_PRETRAINED_SPEC
from aurora_inference.model.loader import _FORBIDDEN_REVISIONS, load_model

_EXPECTED_FILENAMES: dict[ModelName, str] = {
    "aurora-small-pretrained": "aurora-0.25-small-pretrained.ckpt",
    "aurora-pretrained": "aurora-0.25-pretrained.ckpt",
    "aurora-finetuned": "aurora-0.25-finetuned.ckpt",
}
_EXPECTED_HEX_CHARACTERS = string.hexdigits.lower()


def test_aurora_hf_revision_is_pinned_sha() -> None:
    assert AURORA_HF_REVISION not in _FORBIDDEN_REVISIONS
    assert len(AURORA_HF_REVISION) == 40
    assert all(c in _EXPECTED_HEX_CHARACTERS for c in AURORA_HF_REVISION)


@pytest.mark.parametrize("model_name", list(CHECKPOINT_REGISTRY))
def test_resolve_checkpoint_returns_pinned_registry_entry(model_name: ModelName) -> None:
    config = resolve_checkpoint(model_name)

    assert config.model_name == model_name
    assert config.hf_repo_id == AURORA_HF_REPO_ID
    assert config.revision == AURORA_HF_REVISION
    assert config.checkpoint_filename == _EXPECTED_FILENAMES[model_name]


def test_resolve_checkpoint_rejects_unknown_model_name() -> None:
    with pytest.raises(KeyError, match=r"unknown model_name 'not-a-model'"):
        resolve_checkpoint(cast(ModelName, "not-a-model"))


@pytest.mark.parametrize("model_name", list(CHECKPOINT_REGISTRY))
def test_spec_for_model_returns_shared_weather_spec(model_name: ModelName) -> None:
    assert spec_for_model(model_name) is AURORA_PRETRAINED_SPEC


def test_spec_for_model_rejects_unknown_model_name() -> None:
    with pytest.raises(KeyError, match=r"unknown model_name 'not-a-model'"):
        spec_for_model(cast(ModelName, "not-a-model"))


@pytest.mark.parametrize("revision", sorted(_FORBIDDEN_REVISIONS))
def test_load_model_rejects_unpinned_revision(revision: str) -> None:
    with pytest.raises(ValueError, match=r"pinned commit SHA"):
        load_model(revision=revision)


@pytest.mark.parametrize("model_name", list(CHECKPOINT_REGISTRY))
def test_load_model_orchestrates_pinned_ckpt_and_inference_setup(
    model_name: ModelName,
) -> None:
    """Test that load_model() prepares an inference-ready model on the requested device.

    Uses a MagicMock in place of a real Aurora model (_build_model is patched).
    Calls to load_checkpoint(), eval(), and to() are no-ops: they do not download
    from HuggingFace or run PyTorch; they only record that load_model invoked them.
    Assertions inspect that call history (wiring + pinned HF args), not checkpoint bytes.

    Sub-behaviours:
    - build the right model class
    - load checkpoint with pinned HF args
    - call eval()
    - move to device and return
    """
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    # Stub _build_model() so load_model gets fake_model instead of a real Aurora class.
    with patch("aurora_inference.model.loader._build_model", return_value=fake_model) as build:
        result = load_model(model_name, device="cpu")

    # Assert sub-behaviours listed above
    build.assert_called_once_with(model_name)
    fake_model.load_checkpoint.assert_called_once_with(
        repo=AURORA_HF_REPO_ID,
        name=_EXPECTED_FILENAMES[model_name],
        revision=AURORA_HF_REVISION,
    )
    fake_model.eval.assert_called_once_with()
    fake_model.to.assert_called_once_with("cpu")
    assert result is fake_model


def test_load_model_forwards_explicit_pinned_revision_override() -> None:
    """Test that an explicit pinned revision override is forwarded to load_checkpoint.

    Same no-op mock pattern as test_load_model_orchestrates_pinned_ckpt_and_inference_setup:
    fake_model.load_checkpoint() records the call without contacting HuggingFace.
    Proves only that an explicit revision = revision_override is forwarded to load_checkpoint
    """
    revision_override = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch("aurora_inference.model.loader._build_model", return_value=fake_model):
        load_model("aurora-small-pretrained", revision=revision_override)

    fake_model.load_checkpoint.assert_called_once_with(
        repo=AURORA_HF_REPO_ID,
        name=_EXPECTED_FILENAMES["aurora-small-pretrained"],
        revision=revision_override,
    )
