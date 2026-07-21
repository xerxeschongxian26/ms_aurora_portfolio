# Declares targets that are not real files, so `make` always runs their recipes/commands defined here.
.PHONY: install lint format typecheck test test-slow check docker-build docker-run

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

# Build the linux/arm64 CPU image from the Dockerfile (does not run the model).
#   docker build              — execute Dockerfile instructions → image layers
#   --platform linux/arm64    — build for aarch64 Linux (native on M1; matches GH200)
#   --target cpu              — stop at the named "cpu" stage (skip gpu scaffold)
#   -t aurora-inference:cpu   — tag the resulting image as name:tag
#   .                         — build context = current directory (respects .dockerignore)
docker-build:
	docker build --platform linux/arm64 --target cpu -t aurora-inference:cpu .

# Run the image's default CMD (scripts/toy_forward.py) with the host HF cache mounted.
#   docker run                         — create and start a container from an image
#   --rm                               — delete the container filesystem when it exits
#   --platform linux/arm64             — run as aarch64 Linux (must match the image)
#   -e HF_HOME=/cache/huggingface      — tell HuggingFace Hub where to read/write cache
#   -v HOST:CONTAINER                  — bind-mount host cache into the container path
#        $(HOME)/.cache/huggingface    — host side (shared with local / make test-slow)
#        /cache/huggingface            — container side (matches HF_HOME)
#   aurora-inference:cpu               — image to run (no override → uses Dockerfile CMD)
docker-run:
	docker run --rm \
		--platform linux/arm64 \
		-e HF_HOME=/cache/huggingface \
		-v "$(HOME)/.cache/huggingface:/cache/huggingface" \
		aurora-inference:cpu
