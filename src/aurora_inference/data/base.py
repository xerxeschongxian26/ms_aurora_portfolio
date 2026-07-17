"""Batch source seam: swappable producers of Aurora ``Batch`` objects.

Backends (synthetic, ARCO-ERA5, CDS) implement :class:`BatchSource`. Callers
depend only on ``load(init_time, spec)``, not on how bytes were fetched.

Every implementation must call ``validate_input_times`` inside ``load``
(``Batch.metadata.time`` carries only t1). Full ``validate_batch`` runs only at the
inference boundary before ``model.forward``, not inside the ``load(init_time, spec)`` interface
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from aurora import Batch

from aurora_inference.contract import ModelSpec

__all__ = ["BatchSource"]


@runtime_checkable
class BatchSource(Protocol):
    """Structural interface for anything that can produce an inference ``Batch``."""

    def load(self, init_time: datetime, spec: ModelSpec) -> Batch:
        """Load (or synthesize) a batch whose ``metadata.time`` entries are ``init_time`` (t1).

        Implementations must enforce the input timestep with
        ``validate_input_times(t0, init_time, hours=spec.input_timestep_hours)``
        where ``t0 = init_time - timedelta(hours=spec.input_timestep_hours)``.
        """
        ...
