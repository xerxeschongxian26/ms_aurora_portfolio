# syntax=docker/dockerfile:1
# Multi-stage image for aurora-inference.
#
# Platforms (set via `docker build --platform` / `make docker-build PLATFORM=...`):
# - refer to Makefile for more information
#   linux/arm64  — preferred: native for Apple M1 chip; matches Lambda GH200
#   linux/amd64  — native for common cloud GPU hosts (A10 / A100 / H100 on x86_64);
#                  use when GH200 is unavailable. On an arm64 laptop, this build uses
#                  a QEMU (CPU emulator) and is expected to be slower and even error prone

# ---------------------------------------------------------------------------
# cpu — Stage 0 path (build with --platform linux/arm64 or linux/amd64)
# ---------------------------------------------------------------------------

# Base image: Debian Bookworm (a slimmer Linux distribution) with Python 3.12. Names this stage `cpu`
# so `docker build --target cpu` stops here and skips the gpu scaffold.
FROM python:3.12-slim-bookworm AS cpu

# Install uv/uvx from the official Astral image (multi-stage copy; not from repo).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# All subsequent COPY/RUN/CMD paths are relative to /app inside the image.
WORKDIR /app

# uv: precompile bytecode and copy (not symlink) installed files.
# PYTHONUNBUFFERED: stream container logs immediately (no stdout buffering).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependency manifest + lockfile (paths are relative to build context = repo root).
COPY pyproject.toml uv.lock README.md LICENSE ./
# Application package and the synthetic-forward entry script.
COPY src ./src
COPY scripts ./scripts

# Install locked runtime deps into /app/.venv; fail if toml and lock drift (--frozen detects the drift).
RUN uv sync --frozen --no-dev

# Prefer the project venv on PATH so `python` and installed packages resolve there.
ENV PATH="/app/.venv/bin:${PATH}"

# Default container command when `make docker-run` (or `docker run`) is used.
CMD ["python", "scripts/synthetic_forward.py"]

# ---------------------------------------------------------------------------
# gpu; not build by default (`make docker-build` uses cpu stage)
# Base tag is draft for both arches — confirm the platform you need exists:
#   docker manifest inspect nvidia/cuda:12.6.3-runtime-ubuntu22.04
# TODO(stage-2): verify CUDA wheel resolution on the host machine
# ---------------------------------------------------------------------------
# Base image is an Ubuntu 22.04 with CUDA:12.6.3 runtime libraries. No Python by default
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS gpu

# TODO(stage-2): CUDA index (cu124/cu126) for the image platform (linux/arm64 or linux/amd64) —
# blocked on first GPU session on the booked host.
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# installs python 3.12 into the image using uv. This version should match that specified in .toml
RUN uv python install 3.12

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

# Install locked runtime deps into /app/.venv; fail if toml and lock drift (--frozen detects the drift).
RUN uv sync --frozen --no-dev --extra forecast

# Prefer the project venv on PATH so `python` and installed packages resolve there.
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "scripts/real_forecast.py"]
