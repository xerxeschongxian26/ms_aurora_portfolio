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

## Project Development Notes
### Models, Architecture and Mathematical Logic
- The Aurora models appears to consists of smaller sub models for predicting air pollutions (particulate concentration), wave dynamics and of course, weather. Sub categories are also trained at different spacial resolutions
- A 3D Swin Transformer and Perceiver architecture are utilised
- The models are all pretrained and are available in a full and small version. The full version requires 40GB of memory, approximately the capacity of an entry level enterprise GPU such as the NVIDIA RTX H100
- The smaller pretrained versions are provided for debugging purposes. It's input and output signatures (i.e. the shape of the heterogenous inputs and outputs) are identical to the full model but differ only in the complexity of the architectures, which consists of less layers (and also simpler architectures? Not sure!)
- Q - how does the model handle the different inputs, does it map the model choice depending on what has been inputted as a Batch? What happens if I enter an incorrect Batch specification
- Q - how does the 3D Swin Transformer (windowed attention) work? Watch video by Code Emporium, read paper and write primer
- Input Time Step - t-1, and t
- Output time step - t + 1
- The representation resembles a 2nd order Markov property, despite classical numerical methods assuming a 1st order Markov Property
### Data, Data Sources and Data Integrity
- Input data are heterogenous and unnormalised when fed into the network. Normalisation occurs inside the model
- The Batch and Metadata data classes are custom data classes that are part of the Aurora package
- Both the input and output are of dataclass type Batch
- Q - besides the normalisation, what does transformations are applied? Conversion into a Fourier Embedding?
- Q - The docs briefly described a coordinate system that defines polarity of the Lat-Lon coordinates and wind velocity

### Ideas
- Demonstrate the ability to predict the cyclone path (and state?) for the recent Hurrican Melissa (Oct. 2025)
## License

MIT — see [LICENSE](LICENSE).
