"""Load pinned Aurora checkpoints from HuggingFace into eval mode on a device.

``AuroraSmallPretrained`` is for plumbing / debug only — it has no forecast skill.
Never report skill numbers from it. Prefer ``aurora-pretrained`` for evaluation
(Stage 1+).
"""

from __future__ import annotations

import torch
from aurora import Aurora, AuroraPretrained, AuroraSmallPretrained

from aurora_inference.config import (
    AURORA_HF_REVISION,
    ModelName,
    resolve_checkpoint,
)

__all__ = ["load_model"]

_FORBIDDEN_REVISIONS = frozenset({"", "main", "master"})


def load_model(
    model_name: ModelName = "aurora-small-pretrained",
    *,
    device: str | torch.device = "cpu",
    revision: str | None = None,
) -> Aurora:
    """Instantiate ``model_name``, load its pinned checkpoint, return ``eval()`` on ``device``.

    Args:
        model_name: Registry key from :data:`~aurora_inference.config.CHECKPOINT_REGISTRY`.
            Stage 0 default is the small debug checkpoint.
        device: Torch device string or ``torch.device`` (default ``cpu``).
        revision: Optional HF git revision override. Defaults to the pin in
            :data:`~aurora_inference.config.AURORA_HF_REVISION`. Must not be
            ``main`` / ``master`` / empty.

    Returns:
        An :class:`~aurora.model.aurora.Aurora` subclass in eval mode on ``device``.

    Raises:
        KeyError: Unknown ``model_name``.
        ValueError: ``revision`` is unpinned (``main`` / empty).
    """
    config = resolve_checkpoint(model_name)
    pinned_revision = _require_pinned_revision(
        revision if revision is not None else config.revision
    )

    model = _build_model(model_name)
    model.load_checkpoint(
        repo=config.hf_repo_id,
        name=config.checkpoint_filename,
        revision=pinned_revision,
    )
    model.eval()
    return model.to(device)


def _require_pinned_revision(revision: str) -> str:
    """Reject floating branch names; checkpoints must resolve to a commit SHA."""
    if revision in _FORBIDDEN_REVISIONS:
        msg = (
            f"checkpoint revision must be a pinned commit SHA "
            f"(got {revision!r}; expected e.g. {AURORA_HF_REVISION!r}, not 'main')"
        )
        raise ValueError(msg)
    return revision


def _build_model(model_name: ModelName) -> Aurora:
    if model_name == "aurora-small-pretrained":
        return AuroraSmallPretrained()
    if model_name == "aurora-pretrained":
        return AuroraPretrained()
    if model_name == "aurora-finetuned":
        return Aurora()
    # Exhaustiveness guard for future ModelName literals.
    msg = f"no constructor registered for {model_name!r}"
    raise KeyError(msg)
