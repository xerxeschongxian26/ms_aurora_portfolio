# Declares targets that are not real files, so `make` always runs their recipes/commands defined here.
.PHONY: install lint format typecheck test test-slow check docker-build docker-build-cpu docker-build-gpu docker-run docker-run-gpu

# Install the project + dev tools into .venv from the lockfile (reproducible local env).
#   uv sync          — create/update the virtualenv and install dependencies
#   --extra dev      — also install the [project.optional-dependencies] "dev" group
#   --frozen         — do not update uv.lock; install exactly the locked versions
install:
	uv sync --extra dev --frozen

# Lint Python sources (find errors / style issues without changing files).
#   uv run           — run the following tool inside the project .venv
#   ruff check       — Ruff's linter
#   src tests        — paths to lint
lint:
	uv run ruff check src tests

# Auto-format Python sources in place.
#   uv run           — run inside the project .venv
#   ruff format      — Ruff's formatter (like Black)
#   src tests        — paths to format
format:
	uv run ruff format src tests

# Static type-check the package and tests (mypy strict, per pyproject.toml).
#   uv run           — run inside the project .venv
#   mypy             — type checker
#   src tests        — paths to type-check
typecheck:
	uv run mypy src tests

# Run the default (fast) pytest suite.
#   uv run           — run inside the project .venv
#   pytest           — test runner; pyproject addopts excludes `@pytest.mark.slow`
test:
	uv run pytest

# Run only slow/integration tests (may download HF checkpoints).
#   uv run           — run inside the project .venv
#   pytest           — test runner
#   -m slow          — select tests marked with @pytest.mark.slow
test-slow:
	uv run pytest -m slow

# Full local quality gate: lint + types + tests, then verify formatting (no writes).
#   check depends on lint, typecheck, test (Make runs those first)
#   uv run                  — run inside the project .venv
#   ruff format --check     — fail if files need formatting (does not rewrite)
#   src tests               — paths to check
check: lint typecheck test
	uv run ruff format --check src tests

# Declare variables PLATFORM and IMAGE_TAG
# Default value PLATFORM=linux/arm64 (GH200-compatible; native architecture on Apple M1).
# Docker CPU image platform. Preferred default matches GH200 / Apple Silicon.
# For common x86_64 cloud GPUs (A10/A100/H100): use command "make docker-build PLATFORM=linux/amd64"
# Platform string in the tag; `/` character replaced with `-` because Docker tags cannot contain `/`
# - "linux/arm64" has the tag "cpu-linux-arm64" | "linux/amd64" has the tag "cpu-linux-amd64"
PLATFORM ?= linux/arm64
IMAGE_TAG_CPU := aurora-inference:cpu-$(subst /,-,$(PLATFORM))
IMAGE_TAG_GPU := aurora-inference:gpu-$(subst /,-,$(PLATFORM))
# Optional GPU run label: `make docker-run-gpu RUN_TAG=b1-s4 ARGS="--steps 1"`
RUN_TAG ?=
ARGS ?=
ifneq ($(strip $(RUN_TAG)),)
NAME_ARG := --name forecast-$(RUN_TAG)
TAG_ARG := --tag $(RUN_TAG)
else
NAME_ARG :=
TAG_ARG :=
endif

# Build the CPU only image for $(PLATFORM) and names the image with $(IMAGE_TAG_CPU) but does not run it.
#   For x86 cloud hosts: "make docker-build-cpu PLATFORM=linux/amd64"
#   IMAGE_TAG_CPU: aurora-inference:cpu-<platform> (e.g. cpu-linux-arm64)
#   "--target cpu" stops build at the named "cpu" stage and skips the gpu stage
#   docker-build is an alias so README / playbook `make docker-build` still works
docker-build-cpu:
	docker build --target cpu  \
	     		 --platform $(PLATFORM) \
				 -t $(IMAGE_TAG_CPU) .

docker-build: docker-build-cpu

# Build the GPU only image for $(PLATFORM) and names the image with $(IMAGE_TAG_GPU) but does not run it.
#   For x86 cloud hosts: "make docker-build-gpu PLATFORM=linux/amd64"
#   IMAGE_TAG_GPU: aurora-inference:gpu-<platform> (e.g. gpu-linux-arm64)
#   "--target gpu" stops at the gpu stage (independent FROM; does not build cpu)
docker-build-gpu:
	docker build --target gpu  \
	     		 --platform $(PLATFORM) \
				 -t $(IMAGE_TAG_GPU) .

# Run the CPU image default CMD (scripts/synthetic_forward.py) with the host HuggingFace
# cache mounted; deletes the container upon completion
docker-run:
	docker run -e HF_HOME=/cache/huggingface \
			   -v "$(HOME)/.cache/huggingface:/cache/huggingface" \
			   --rm \
			   --platform $(PLATFORM) \
			   $(IMAGE_TAG_CPU)

# Run the GPU image (scripts/real_forecast.py) with host GPUs, HF cache, and
# ./outputs mounted so logs/PNGs survive --rm. Pass RUN_TAG and/or ARGS.
#   make docker-run-gpu PLATFORM=linux/amd64 RUN_TAG=b1-s4 ARGS="--steps 1"
docker-run-gpu:
	mkdir -p outputs
	docker run --gpus all \
			   -e HF_HOME=/cache/huggingface \
			   -v "$(HOME)/.cache/huggingface:/cache/huggingface" \
			   -v "$(CURDIR)/outputs:/app/outputs" \
			   $(NAME_ARG) \
			   --rm \
			   --platform $(PLATFORM) \
			   $(IMAGE_TAG_GPU) \
			   python scripts/real_forecast.py $(TAG_ARG) $(ARGS)
