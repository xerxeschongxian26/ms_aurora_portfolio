# ADR005: Evaluation approach

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-19 |
| **Related** | [`docs/benchmark-target.md`](../benchmark-target.md); [ADR 0003](0003-checkpoint-pinning.md); [ADR 0004](0004-hres-t0-source-of-record.md); `evaluation/metrics.py`, `grids.py`, `fidelity.py`, `variants.py` |

Sections follow the [Backstage ADR template](https://github.com/backstage/backstage/blob/master/docs/architecture-decisions/adr000-template.md) (Context / Decision / Consequences; Michael Nygard).

## Context

Stage 3 will change how we *run* Aurora (fewer bits; later maybe batching).
That is a different question from “is this still the paper model?”. We keep
those as two checks:

| Check | Question | Compared to |
|---|---|---|
| **Q1 — skill** | Does this pipeline match the published 0.25° fine-tune? | WeatherBench2 (WB2) RMSE vs HRES-T0 |
| **Q2 — fidelity** | Did a cheaper run change the maps? | Saved fp32 forecasts, and HRES-T0 again |

An earlier plan was to import the `weatherbench2` package in `src/` and wrap
it with `converter.py` and `skill.py`. That library is built for research
notebooks (ACC, climatology, WB2 names), not for this service. It would also
push us to keep a full 10-day cube in memory. We already score one forecast
hour at a time and throw the tensors away. A second layout (WB2 names, lat
order, stacked lead times) is an easy place for a quiet bug.

Q2 has two standard questions, not one. CESM-ECT (Baker et al., 2015)
compares a new run to a trusted baseline and does **not** use ground truth.
Quantization papers compare each run to **truth**. A draft here divided
those into one “fidelity ratio”. Neither community does that, so we dropped
it.

Init dates in Python (`BENCHMARK_INITS_*`) would drift from the TOML files
and from the hole list. The paper uses 730 starts; we cannot afford that.
Six starts are cheap enough to try every change, and too few to publish.

Measured floors and the SKU are in
[`docs/benchmark-target.md`](../benchmark-target.md) §6–§11. This ADR locks
the method, not those numbers.

## Decision

Q1 and Q2 are separate scripts: `eval_rmse.py` and `run_fidelity.py`. We
will not add a third name (`run_benchmark.py`).

Latitude-weighted MSE lives in `aurora_inference.evaluation.metrics.MSE`
(comment points at WB2; fixture
`tests/fixtures/eval_z500_rmse_on_cropped_grid.npz`). Runtime `src/` does
not import `weatherbench2`. That package stays a `notebooks` extra for
plots. ACC and climatology wait (`TODO(stage-4)`).

We convert **one lead at a time** in `evaluation/grids.py`: `Batch` →
`xarray` (south-to-north lat), score, drop. No `converter.py`. No stacked
lead-time Dataset. Field names stay Aurora short names (`2t`, `z`, …). Disk
layout is `baselines/init-{id}/lead-{HHH}.zarr`.

Start times live in TOML (`hres_t0_2022_spread_rollout.toml`,
`hres_t0_2022_fidelity_screen_rollout.toml`). `build_eval_schedule` allows
only 2022 at 00:00 and 12:00 UTC. No `BENCHMARK_INITS_*` in Python. We skip
any window that hits `configs/hres_t0_outsample_holes.csv`. A hole is a
store time with non-finite values.

Q1 uses **30** hole-aware starts through 2022, not 730. PASS is our rule:
every headline variable, every lead, within 5% of WB2’s n=730 curve. The 5%
band is not a paper number. Weights: [ADR 0003](0003-checkpoint-pinning.md).
Truth: [ADR 0004](0004-hres-t0-source-of-record.md).

Q2 uses two subsets of those 30. **Screen (n=6)** is the first try on every
change. **Confirm (n=30)** is required before RMSE goes in a post, the
README, or a report. Always write `n`. Do not add dates to the screen file.
Do not publish screen RMSE as the result.

Q2 stores **two RMSEs**, both using the same `MSE`:

- **M1** — variant vs saved fp32 maps (`compute_rmse_score_variant_vs_baseline`).
  Did the numerics move? Same idea as CESM-ECT (Baker et al., 2015).
- **M2** — variant vs HRES-T0 (`compute_rmse_score_variant_vs_ground_truth`).
  Did skill get worse? Same question as Q1.

There is no `skill_penalty` in `src/`. If we want a percent change vs the
baseline’s skill, we compute it later from the two tables. Dividing M1 by
skill error (the old “fidelity ratio”) stays dropped.

Stage 3 adds a row to `VARIANTS` in `evaluation/variants.py`. The Q2 script
looks up the name and builds the model. Stage 2 only ships `fp32-baseline`.
Torch flags are part of that baseline. We record what torch actually had,
not what the variant text claimed.

We will not rebuild `converter.py` or `skill.py` unless a later work package
asks.

## Consequences

- Default tests stay offline. They cover MSE, grids, the two RMSE helpers,
  the variant list, and the schedule guards. They do not download WB2 or
  weights.
- A `weatherbench2` upgrade cannot quietly change Q1. Notebooks may still
  use the package; that is not the gate.
- Scoring holds one lead in RAM, not forty. Later jobs read the saved
  headline maps; they do not re-run fp32 to score.
- Changing the baseline (matmul precision, `cudnn.benchmark`,
  `cudnn.allow_tf32`) means a new archive and new timings.
- Screen numbers stay internal. Public RMSE is n=30, with the GPU named. A
  tiny M1 at or below the same-run floor is noise.
- A new cheaper run is a registry row plus `run_fidelity.py`, not a new
  metric library. Training, QAT, and LoRA tuning stay out
  ([ADR 0001](0001-inference-only-scope.md)).
- Using a small initial-condition ensemble as the M1 pass bar is
  `TODO(stage-4)`.

## References

- Baker, A. H. et al. (2015). A new ensemble-based consistency test for the
  Community Earth System Model. *Geosci. Model Dev.* 8, 2829–2840.
  <https://doi.org/10.5194/gmd-8-2829-2015>
- [`docs/benchmark-target.md`](../benchmark-target.md)
- [microsoft/aurora#197](https://github.com/microsoft/aurora/issues/197)
