# Benchmark target — Q1 skill gate (Stage 2)

The published skill curves live in supplementary section H of the Aurora paper
(Bodnar et al., 2025). **Fig. H7** is RMSE vs lead (to 10 days) for the **eight
headline variables**, with both **unnormalised** (native units) and
**normalised** (GraphCast 0.25° as baseline) rows. Figures H8 and H9 are
additional atmosphere panels, not a second Q1 gate.

## 1. Model / data / reference

| Role | This repo |
|---|---|
| **Model** | Aurora 0.25° fine-tuned (`aurora-finetuned` → `aurora-0.25-finetuned.ckpt`) |
| **Data / truth** | WB2 HRES-T0 analysis at **init + lead**, from the same local splice used as input. Not ERA5. |
| **Reference** | WB2 Aurora vs HRES-T0 2022 global **unnormalised** RMSE (**n=730**) for the **8** Fig. H7 headline variables. Numeric oracle is the WB2 table, not a digitised PNG. The GraphCast-normalised H7 rows are not the Q1 gate. |

H8 (lower atmosphere) and H9 (upper atmosphere) are context only — they are
not a second pass criterion.

## 2. Target curves

<p align="center">
  <img src="images/supinfo_figure_h7_rmse_headline_vars.png" alt="Figure H7: RMSE of the eight headline variables, unnormalised and GraphCast-normalised" width="650"/>
</p>

Fig. H7’s eight headline fields (2T, 10U, MSL, U500, Z500, T500, T850, Q500)
are this repo’s Q1 set. Native 2-D Aurora slices — not WB2 Table 3 (no precip,
no `10v`). Q1 scores the **unnormalised** RMSE curves against WB2 Aurora vs
HRES-T0 truth, not the GraphCast-normalised rows.

<p align="center">
  <img src="images/supinfo_figure_h8_rmse_lower_atmosphere.png" alt="Figure H8: Absolute RMSE – lower atmosphere" width="650"/>
</p>
<p align="center">
  <img src="images/supinfo_figure_h9_rmse_upper_atmosphere.png" alt="Figure H9: Absolute RMSE – upper atmosphere" width="650"/>
</p>

## 3. Eval protocol (this repo)

The **paper** protocol is every 00/12 UTC init in 2022 (**730** inits) × 40
six-hour leads (1–10 days). This repo does **not** run 730.

Stage 2 Q1 uses a **hole-aware year-spread of n=30** 00/12 UTC inits in 2022,
40 steps (`t+6h` … `t+240h`). The named list is
[`configs/hres_t0_2022_spread_rollout.toml`](../configs/hres_t0_2022_spread_rollout.toml)
(TOML is the source of record; there is no `BENCHMARK_INITS_*` in Python).
Windows that touch
[`configs/hres_t0_outsample_holes.csv`](../configs/hres_t0_outsample_holes.csv)
are skipped. Late-December inits need truth into early January 2023; the last
spread lead is `2023-01-08T00`, inside the WB2 store tail (`2023-01-10T18`).

| Knob | Paper | This repo (Q1) |
|---|---|---|
| Year | 2022 | 2022 |
| Init hours | 00 and 12 UTC | 00 and 12 UTC |
| n inits | 730 | **30** |
| Leads | 40 × 6 h | 40 × 6 h |
| Variables | Fig. H7: 8 headline (unnormalised rows) | Same 8: `2t`, `10u`, `msl`, `u500`, `z500`, `t500`, `t850`, `q500` |
| Metric | WB2 lat-weighted RMSE (and ACC) | Lat-weighted RMSE only |

**Metric.** Latitude-weighted RMSE, `cos(lat)` weights normalised to unit mean
(WB2 / Aurora Supp. F, eq. F14), implemented in
`aurora_inference.evaluation.metrics.MSE`. ACC and WB2 climatology are **not**
computed (`TODO(stage-4)`). Runtime `src/` does not import `weatherbench2`.

## 4. Honesty guards

**Checkpoint identity (confirmed).** Q1 loads `aurora-0.25-finetuned.ckpt` at
Hugging Face revision `0be7e57c685dac86b78c4a19a3ab149d13c6a3dd`
(`AURORA_HF_REVISION`; [ADR 0003](decisions/0003-checkpoint-pinning.md)). These
weights are the 0.25° HRES-T0 fine-tune trained on **2016–2021** with **2022
held out**, and they are the model behind supplementary Fig. H7. Source: paper
Table C4, plus maintainer confirmation in
[microsoft/aurora#197](https://github.com/microsoft/aurora/issues/197)
(Wessel Bruinsma, 2026-08-19). This does **not** apply to the 0.1° fine-tune,
which saw 2022; this repo does not load that checkpoint for Q1.

**2022-only / 00-12 UTC (enforced in code).**
`build_eval_schedule` rejects any init outside 2022 or not on the hour at 00 or
12 UTC (`EVAL_YEAR`, `ALLOWED_INIT_HOURS` in
`aurora_inference.evaluation.evaluation_schedule`). The guard sits on the
schedule, not inside `HresT0Source` — the store also carries the 2016–2021
fine-tuning years, and a stray training-year init would be silent eval-on-train.
The previous 6-hour input (`t−6h`) may fall on `2021-12-31`; that is allowed.

## 5. Pass criterion

**Definition (this project, not a paper number).** For each headline variable
and each lead τ in `+6 h … +240 h`:

```
δ(τ) = ( RMSE_n=30(τ) − RMSE_WB2(τ) ) / RMSE_WB2(τ)
```

PASS requires `|δ(τ)| < 0.05` at **every** compared lead, for **every**
variable in the set. A mean `|δ|` under 5% with a spike above 5% is a fail.
Because the released checkpoint is the paper checkpoint, the curves should
coincide, not merely rhyme.

**Headline set (8)** — Fig. H7 fields, Aurora short names, native 2-D slices.
Q1 uses **unnormalised** RMSE only:

| Key | Field |
|---|---|
| `2t` | 2 m temperature |
| `10u` | 10 m u-component of wind |
| `msl` | mean sea-level pressure |
| `u500` | u-component of wind at 500 hPa |
| `z500` | geopotential at 500 hPa |
| `t500` | temperature at 500 hPa |
| `t850` | temperature at 850 hPa |
| `q500` | specific humidity at 500 hPa |

This is **not** WB2 Table 3 (no precipitation, no `10v`). Overlay and `|δ|`
panels: gitignored `notebooks/_scratch/stage2_eval_rmse.ipynb`.

**Result.** **PASS** (n=30 hole-aware 2022 spread vs WB2 n=730). `|δ(τ)|`
stayed below 5% at every lead for all eight variables. README figure is WP6.

## References

- [`docs/hres-t0-source-notes.md`](hres-t0-source-notes.md)
- [`docs/decisions/0003-checkpoint-pinning.md`](decisions/0003-checkpoint-pinning.md)
- [`configs/hres_t0_2022_spread_rollout.toml`](../configs/hres_t0_2022_spread_rollout.toml)
- [microsoft/aurora#197](https://github.com/microsoft/aurora/issues/197)
- [`.cursor/plans_archived/stage1_forecastpipeline_completed.md`](../.cursor/plans_archived/stage1_forecastpipeline_completed.md)
