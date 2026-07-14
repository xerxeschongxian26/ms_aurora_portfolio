.PHONY: install lint format typecheck test test-slow check docker-build docker-run

install:
	uv sync --extra dev --frozen

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

check: lint typecheck test
	uv run ruff format --check src tests

docker-build:
	docker build --platform linux/arm64 --target cpu -t aurora-inference:cpu .

docker-run:
	docker run --rm aurora-inference:cpu
