# ms_aurora_portfolio

Machine learning portfolio project for MS Aurora.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --all-extras
uv run pre-commit install
```

For long notebook sessions, activate the virtual environment:

```bash
source .venv/bin/activate
```

**Pip fallback** (no uv): export locked requirements with `uv export --extra dev -o requirements.txt`, then `pip install -r requirements.txt`. Editable install (`pip install -e .`) becomes available once you add a `src/` package layout.

## How to train

Training scripts are not yet defined. Add entrypoints under `scripts/` and configs under `configs/`.

## How to run tests

```bash
uv run pytest -q
```

## How to type-check

Once Python modules exist (e.g. under `src/` or `scripts/`), run:

```bash
uv run mypy <path>
```

## Notebooks

- Prototype in `notebooks/` (outputs stripped by nbstripout).
- Use `notebooks/_scratch/` for local-only scratch pads (gitignored).
- When the project layout is decided, promote stable code into a package or module tree.

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| —      | —     | —     |

## License

MIT — see [LICENSE](LICENSE).
