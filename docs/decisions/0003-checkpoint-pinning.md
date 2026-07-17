# 0003 — Checkpoint pinning

**Status:** Accepted
**Date:** 2026-07-17
**Related:** WP4 (`config.py`, `model/loader.py`); package pin in `pyproject.toml`

## Context

Aurora checkpoints live on HuggingFace (`microsoft/aurora`) and are loaded by the
`microsoft-aurora` Python package. The upstream repo is actively developed: the
Aurora 1.5 work changed the `Batch` contract (e.g. `lead_times` moved out of
`Batch`), added variables, expanded static fields, and changed lead-time
embeddings. Loading from the floating `main` (or `master`) branch means checkpoint
bytes can change between runs without a code change. Benchmark and plumbing
numbers would then drift silently.

The installed package also embeds default HF revision SHAs
(`default_checkpoint_revision`). Relying on that library default alone is fragile:
an upgrade of `microsoft-aurora` can change the default SHA, and an explicit pin
in *our* config makes the reproducibility contract visible in this repo.

## Decision

1. Pin `microsoft-aurora==1.8.0` exactly in `pyproject.toml` (no version range).
2. Pin the HuggingFace git revision explicitly in `src/aurora_inference/config.py`
   as `AURORA_HF_REVISION`, and pass that revision into `load_checkpoint` — do
   not rely on the library default or on `main`.
3. Forbid floating revisions (`main`, `master`, empty string) in `load_model`.

**Pinned values (must move together):**

| Field | Value |
|---|---|
| Package | `microsoft-aurora==1.8.0` |
| HF repo | `microsoft/aurora` |
| HF revision (commit SHA) | `0be7e57c685dac86b78c4a19a3ab149d13c6a3dd` |

That SHA is the `default_checkpoint_revision` shipped with `microsoft-aurora==1.8.0`
for both `AuroraSmallPretrained` and `AuroraPretrained`. It is not an arbitrary
HF tip commit — it keeps package code and checkpoint bytes in lockstep.

Cache location remains the HuggingFace default (`~/.cache/huggingface/`).
Checkpoint files are never committed.

## Consequences

- Reproducible loads: the same package + SHA always resolve the same weight bytes
  (modulo HF availability).
- Upgrades require a deliberate pair of changes: bump `microsoft-aurora` in
  `pyproject.toml` / `uv.lock`, update `AURORA_HF_REVISION` to the new package’s
  default (or a reviewed SHA), and update this ADR.
- CI and default pytest never download checkpoints; real downloads stay behind
  `@pytest.mark.slow` (WP5+).
- Callers cannot opt into `main` via `load_model(..., revision="main")`; overrides
  must still be a commit SHA.
