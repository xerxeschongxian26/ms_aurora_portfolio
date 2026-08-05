"""Pinned HuggingFace checkpoint configuration for Aurora models.

Never load from the floating ``main`` branch of ``microsoft/aurora``. Checkpoint
bytes and the ``microsoft-aurora`` package version must move together — see
``docs/decisions/0003-checkpoint-pinning.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aurora_inference.contract import AURORA_PRETRAINED_SPEC, ModelSpec

__all__ = [
    "AURORA_HF_REPO_ID",
    "AURORA_HF_REVISION",
    "CHECKPOINT_REGISTRY",
    "CheckpointConfig",
    "ModelName",
    "resolve_checkpoint",
    "spec_for_model",
]

# HF git commit SHA matching microsoft-aurora==1.8.0 defaults. Do not use "main".
AURORA_HF_REPO_ID = "microsoft/aurora"
AURORA_HF_REVISION = "0be7e57c685dac86b78c4a19a3ab149d13c6a3dd"  # pragma: allowlist secret

ModelName = Literal["aurora-small-pretrained", "aurora-pretrained", "aurora-finetuned"]


@dataclass(frozen=True)
class CheckpointConfig:
    """Resolved HF location for one Aurora checkpoint variant."""

    model_name: ModelName
    checkpoint_filename: str
    hf_repo_id: str = AURORA_HF_REPO_ID
    revision: str = AURORA_HF_REVISION


CHECKPOINT_REGISTRY: dict[ModelName, CheckpointConfig] = {
    "aurora-small-pretrained": CheckpointConfig(
        model_name="aurora-small-pretrained",
        checkpoint_filename="aurora-0.25-small-pretrained.ckpt",
    ),
    "aurora-pretrained": CheckpointConfig(
        model_name="aurora-pretrained",
        checkpoint_filename="aurora-0.25-pretrained.ckpt",
    ),
    "aurora-finetuned": CheckpointConfig(
        model_name="aurora-finetuned",
        checkpoint_filename="aurora-0.25-finetuned.ckpt",
    ),
}


def resolve_checkpoint(model_name: ModelName) -> CheckpointConfig:
    """Return the pinned checkpoint config for ``model_name``.

    Raises:
        KeyError: If ``model_name`` is not in :data:`CHECKPOINT_REGISTRY`.
    """
    try:
        return CHECKPOINT_REGISTRY[model_name]
    except KeyError as exc:
        known = ", ".join(sorted(CHECKPOINT_REGISTRY))
        msg = f"unknown model_name {model_name!r}; expected one of: {known}"
        raise KeyError(msg) from exc


def spec_for_model(model_name: ModelName) -> ModelSpec:
    """Return the :class:`~aurora_inference.contract.ModelSpec` for ``model_name``.

    Stage 0 weather checkpoints share :data:`~aurora_inference.contract.AURORA_PRETRAINED_SPEC`.
    AirPollution / Wave specs are Stage 5+.
    """
    resolve_checkpoint(model_name)  # validate name early
    return AURORA_PRETRAINED_SPEC
