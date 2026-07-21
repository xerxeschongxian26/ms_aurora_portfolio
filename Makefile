# Declares targets that are not real files, so `make` always runs their recipes/commands defined here.
.PHONY: install lint format typecheck test test-slow check docker-build docker-run

# Docker CPU image platform. Preferred default matches GH200 / Apple Silicon.
# For common x86_64 cloud GPUs (A10/A100/H100): make docker-build PLATFORM=linux/amd64
PLATFORM ?= linux/arm64
# Full platform in the tag; `/` → `-` because Docker tags cannot contain `/`
# (linux/arm64 → cpu-linux-arm64, linux/amd64 → cpu-linux-amd64).
IMAGE_CPU := aurora-inference:cpu-$(subst /,-,$(PLATFORM))

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

# Build the CPU image for $(PLATFORM) (does not run the model).
#   Default PLATFORM=linux/arm64 (GH200-faithful; native on M1).
#   x86 cloud hosts: make docker-build PLATFORM=linux/amd64
#   Tags: aurora-inference:cpu-<platform> and aurora-inference:cpu (latest build for PLATFORM)
#   --target cpu — stop at the named "cpu" stage (skip gpu scaffold)
docker-build:
	docker build --platform $(PLATFORM) --target cpu \
		-t $(IMAGE_CPU) -t aurora-inference:cpu .

# Run the image's default CMD (scripts/toy_forward.py) with the host HF cache mounted.
#   PLATFORM must match the image that was built.
#   Override: make docker-run PLATFORM=linux/amd64
docker-run:
	docker run --rm \
		--platform $(PLATFORM) \
		-e HF_HOME=/cache/huggingface \
		-v "$(HOME)/.cache/huggingface:/cache/huggingface" \
		$(IMAGE_CPU)
