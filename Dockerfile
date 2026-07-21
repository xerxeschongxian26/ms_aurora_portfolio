# syntax=docker/dockerfile:1
# Multi-stage image for aurora-inference.
# Default build target is `cpu` (Stage 0). GPU is scaffolded for Stage 1.
#
# Platforms (set via `docker build --platform` / `make docker-build PLATFORM=...`):
#   linux/arm64  — preferred: native on Apple Silicon; matches Lambda GH200 (Grace + H100)
#   linux/amd64  — common cloud GPU hosts (A10 / A100 / H100 on x86_64); use when GH200
#                  is unavailable. On an arm64 laptop this build uses QEMU and is slower.
#
# Same Dockerfile stages for both; do not fork amd64 vs arm64 recipes.

# ---------------------------------------------------------------------------
# cpu — Stage 0 path (build with --platform linux/arm64 or linux/amd64)
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
# Base tag is draft for both arches — confirm the platform you need exists:
#   docker manifest inspect nvidia/cuda:12.6.3-runtime-ubuntu22.04
# TODO(stage-1): verify CUDA wheel resolution on the booked host —
#   blocked on first GPU session (aarch64/cu12x on GH200; amd64/cu12x on x86 H100/A100)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS gpu

# TODO(stage-1): install Python 3.12 + uv, then resolve torch from the PyTorch
# CUDA index (cu124/cu126) for the image platform (linux/arm64 or linux/amd64) —
# blocked on first GPU session on the booked host.
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
