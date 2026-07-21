# syntax=docker/dockerfile:1
# Multi-stage image for aurora-inference.
# Default build target is `cpu` (Stage 0). GPU is scaffolded for Stage 1.

# ---------------------------------------------------------------------------
# cpu — verified Stage 0 path (linux/arm64 via `docker build --platform`)
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS cpu

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "scripts/toy_forward.py"]

# ---------------------------------------------------------------------------
# gpu — scaffold only; do not build by default (`make docker-build` uses cpu)
# Verify the base image has a linux/arm64 manifest before locking the tag:
#   docker manifest inspect nvidia/cuda:12.6.3-runtime-ubuntu22.04
# TODO(stage-1): verify aarch64 CUDA wheel resolution on GH200 — blocked on first GPU session
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS gpu

# TODO(stage-1): install Python 3.12 + uv, then resolve torch from the PyTorch
# CUDA index (cu124/cu126) for aarch64 — blocked on first GPU session on GH200.
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

# Placeholder: Stage 1 replaces this with CUDA-index torch install + uv sync.
RUN echo "GPU stage is scaffold only; build with --target cpu for Stage 0" >&2 \
    && exit 1

CMD ["python", "scripts/toy_forward.py"]
