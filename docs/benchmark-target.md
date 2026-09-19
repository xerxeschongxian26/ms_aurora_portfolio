# Benchmark target — Q1 skill and Q2 floor (Stage 2)

The published skill curves live in supplementary section H of the Aurora paper
(Bodnar et al., 2025). **Fig. H7** is RMSE vs lead (to 10 days) for the **eight
headline variables**, with both **unnormalised** (native units) and
**normalised** (GraphCast 0.25° as baseline) rows. Figures H8 and H9 are
additional atmosphere panels, not a second Q1 gate.

Q1 (§1–§5) is the skill gate: does this pipeline match WeatherBench2? Q2
(§6–§11) is the instrument: what *is* the fp32 baseline, how noisy is the
meter, and how later runs must report `n`.

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
are skipped. A hole is a store time found to contain non-finite values.
Late-December inits need truth into early January 2023; the last
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
stayed below 5% at every lead for all eight variables. Overlay:
[`docs/images/baseline_30inits_rmse_vs_lead_time.png`](images/baseline_30inits_rmse_vs_lead_time.png)
(also in the README).

## 6. Baseline definition (`fp32-baseline`)

Every later speed or memory claim is relative to this configuration. Changing
it means re-timing every row. The name lives in
[`src/aurora_inference/evaluation/variants.py`](../src/aurora_inference/evaluation/variants.py).
Flags below are **read back** from `torch.backends.*` /
`torch.get_float32_matmul_precision()` into `session.json` via
`aurora_inference.runlog.read_hygiene_flags` — not copied from the variant
description.

**Job that saves the maps:** `scripts/save_baseline_forecasts.py` (git
`b2182f8`, tag `wp5b-baselines-n30`, `--fields full` then CPU-trimmed to the
eight Q1 slices). **Job that scores a later run against them:**
`scripts/run_fidelity.py`. `scripts/eval_rmse.py` is the Q1 skill job; it
does **not** set matmul precision or `cudnn.benchmark`.

| Knob | Value (GPU `session.json`) | Notes |
|---|---|---|
| SKU | NVIDIA A100-SXM4-40GB | Lambda; driver `580.126.20`; CUDA runtime `13.0`; `torch 2.12.1+cu130` |
| Checkpoint | `aurora-0.25-finetuned.ckpt` @ `0be7e57` | `microsoft-aurora==1.8.0`; LoRA left on |
| `Aurora(autocast=…)` | not passed (`False`) | Encoder, backbone, and decoder stay fp32 |
| `float32_matmul_precision` | `"highest"` | True fp32 matmul; TF32 matmul **off** |
| `cuda_matmul_allow_tf32` | `false` | Matches `"highest"` |
| `cudnn_allow_tf32` | `true` | **Not set by us** — Ampere/PyTorch default. Convs may use TF32; matmul does not. |
| `cudnn_benchmark` | `true` | cuDNN times conv algorithms and keeps the fastest. Can pick a different winner **across jobs**. |
| `inference_mode_enabled` | `false` in JSON | Probe at `collect_env()`, outside `run_rollout`'s `with`. Forecasts still used `torch.inference_mode()`. |
| `offload_to_cpu` | `true` | Peak VRAM is model + one step |
| Warm-up | 3 discarded 1-step forwards | Timing / autotune only; not in the archive |
| Batch | `B=1` | fp32 `B=2` OOMs this 40 GB card |

**Archive (source of record).** Headline zarrs on NFS:

`/home/ubuntu/xerxes-nfs/aurora-baselines/wp5b-baselines-n30-headline/`

Layout `baselines/init-{id}/lead-{HHH}.zarr`. Pairing key is
`(init_id, lead_hours)` with `init_id` matched by **`init_time`**, not by
screen ordinal 1..6. Full-field cubes (`…/wp5b-baselines-n30/`) were written
first, trimmed 1 200/1 200 allclose, then deleted. Do not re-persist the 30
unless this definition changes (`cudnn.allow_tf32` or `benchmark=False`
would be a new archive).

**Cost of this baseline on the recorded SKU** (persist `run_init-1.json`;
same allocated peak on every persist init): mean step wall ~6.3 s after
warm-up; `max_memory_allocated` **29105211392 B (~27.1 GiB)**;
`max_memory_reserved` **40336621568 B (~37.6 GiB)**; host RSS ~26 GiB.

## 7. Kernel profile (D9)

One forward after warm-up, `torch.profiler` (CPU + CUDA, `record_shapes=True`),
exported as `trace.json` (gitignored; not committed). Written 2026-09-17
15:54:45 UTC; profiled-step wall **6.30 s**. Same persist job as §6.

**Verdict.** The step is **matmul-bound on CUDA cores**, not memory-bound
and not launch-bound. Kernel CUDA time sums to ~6.06 s (~96% of wall).
Top kernels by self CUDA time:

| Kernel | Share of kernel time | Launches |
|---|---|---|
| `ampere_sgemm_128x64_tn` | 82.8% | 307 |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm80` (fp32 memory-efficient attention) | 4.6% | 50 |
| `vectorized_elementwise` add | 3.2% | 198 |
| `GeluCUDAKernelImpl` | 1.5% | 54 |
| unary `vectorized_elementwise` | 1.4% | 110 |

All GEMM names together are ~84%. Copies and permutes are ~1%. The only
TF32 tensor-core name is a tiny
`sm80_xmma_fprop_implicit_gemm_tf32f32…` (~0.07%), which matches
**cuDNN TF32 on, matmul TF32 off**. A later `tf32` row
(`float32_matmul_precision="high"`) is aimed at this sgemm bottleneck;
flipping `cudnn.allow_tf32` would be a no-op on this trace.

## 8. Measurement floor (D10)

Two identical `fp32-baseline` 40-step rollouts in **one process**, init 1
(`2022-01-01T12`), `offload_to_cpu=True`, `cudnn.benchmark=True`. Paired
RMSE at leads **24 / 120 / 240 h** over the full persist field set
(9 weather variables × levels).

| Item | Result |
|---|---|
| SKU | NVIDIA A100-SXM4-40GB |
| `max_rmse` | **0.0** (207 rows; `session.json` `floor.warnings` empty) |
| Rollout wall | **252.64 s** + **252.78 s** |

This is the noise floor **inside one job** (same cached conv winner). It is
not a guarantee that a **new** process will bit-match last week's zarrs —
`cudnn.benchmark` may pick a different algorithm on the next launch
([PyTorch CUDA convolution benchmarking](https://docs.pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking)).
§9 is the cross-job check against the saved maps. An M1 value at or below
this floor is noise, not a speed-up.

## 9. Q2 screen floor (`fp32-baseline` vs archive)

New GPU job (`scripts/run_fidelity.py`, git `5945d60`, 2026-09-18) vs the
**headline** NFS archive in §6. Variant name `fp32-baseline`. Screen TOML
[`configs/hres_t0_2022_fidelity_screen_rollout.toml`](../configs/hres_t0_2022_fidelity_screen_rollout.toml)
(n=6). Hygiene readback matches §6.

Pairing used persist `init_id` by `init_time` (spread ids **1, 6, 15, 16,
22, 24**), not screen ordinal 1..6:

| Persist `init_id` | `init_time` |
|---|---|
| 1 | `2022-01-01T12` |
| 6 | `2022-04-07T00` |
| 15 | `2022-07-18T12` (heat-dome) |
| 16 | `2022-07-25T12` |
| 22 | `2022-09-27T00` (Hurricane Ian) |
| 24 | `2022-10-24T12` |

**vs-baseline:** 1 920 / 1 920 rows at RMSE **0** (`6 × 40 × 8` headline
slices; each init `max_rmse_variant_vs_baseline=0`). That is the Q2 wiring
floor, not skill. **vs-truth** tables are real forecast error against
HRES-T0 — do not treat them as a zero-check.

Mean CUDA step ~6.3 s; peak allocated **~27.1 GiB** / reserved **~37.6 GiB**
(same byte counts as persist). Wall ~38 min (10:05:27–10:43:41 UTC).
Forecast zarrs were **not** re-written. This is **not** an n=30 confirm of
fp32 vs itself; do not re-run that.

## 10. Screen and confirm tiers

Q2 uses two named subsets of the n=30 spread. Both are hole-aware. Both
are already in the splice. TOML is the source of record; do not hardcode
init lists in Python. Do not grow the screen file.

| Name | n | TOML | Use |
|---|---|---|---|
| Screen | **6** | `configs/hres_t0_2022_fidelity_screen_rollout.toml` | First pass on every variant. Cheap. Catches large numerical divergence. |
| Confirm | **30** | `configs/hres_t0_2022_spread_rollout.toml` | Required before any Q2 RMSE appears in a post, README, or report. |

The screen is one init per named month (Jan / Apr / Jul / Oct) plus two
high-gradient cases (European heat-dome `2022-07-18T12`; Hurricane Ian
`2022-09-27T00`). Variant-vs-baseline is paired (same init, weights, and
data path), so six inits catch catastrophes; they do **not** bound the tail
or decide a marginal variant. Precision error is state-dependent and grows
with lead. **State `n` next to every number.** Paper n=730 stays out of
Stage 2.

## 11. Two Q2 RMSEs

Two questions, two numbers. Report both. Neither replaces the other. Code:
`aurora_inference.evaluation.fidelity` (keyword-only wrappers over the same
`MSE` as Q1). There is no `skill_penalty` function in `src/`.

**M1 — consistency: variant vs baseline.**

```
RMSE(variant, baseline)
```

No ground truth. Answers *did the numerics change the model?* Precedent:
CESM Ensemble Consistency Test (CESM-ECT; Baker et al., 2015, *Geosci.
Model Dev.* 8, 2829–2840). M1 is the sensitive signal: a field can move
before skill vs truth moves. D10 and the n=6 screen rank on M1.

**M2 — skill: variant vs HRES-T0.**

```
RMSE(variant, truth)
```

Answers *did the forecast get worse?* Same lat-weighted RMSE as Q1.
Optional percentage penalty

```
[ RMSE(variant, truth) − RMSE(baseline, truth) ] / RMSE(baseline, truth)
```

is **post-processing** from two RMSE tables, not a library primitive.

**Retracted.** An earlier draft divided M1 by the baseline's skill error to
make one "fidelity ratio". That composite has no precedent in CESM-ECT
(which does not normalise against skill) or in the quantization convention
(which has no variant-vs-baseline term). It is dropped. Cite or drop:
a number without its floor, its `n`, or its SKU is a reporting failure.

A perturbed-IC spread as a citable M1 tolerance (CESM-ECT's deeper idea)
is `TODO(stage-4)`.

## References

- [`docs/hres-t0-source-notes.md`](hres-t0-source-notes.md)
- [`docs/decisions/0003-checkpoint-pinning.md`](decisions/0003-checkpoint-pinning.md)
- [`docs/decisions/0004-hres-t0-source-of-record.md`](decisions/0004-hres-t0-source-of-record.md)
- [`configs/hres_t0_2022_spread_rollout.toml`](../configs/hres_t0_2022_spread_rollout.toml)
- [`configs/hres_t0_2022_fidelity_screen_rollout.toml`](../configs/hres_t0_2022_fidelity_screen_rollout.toml)
- [microsoft/aurora#197](https://github.com/microsoft/aurora/issues/197)
- Baker, A. H. et al. (2015). A new ensemble-based consistency test for the
  Community Earth System Model. *Geosci. Model Dev.* 8, 2829–2840.
  <https://doi.org/10.5194/gmd-8-2829-2015>
- [`.cursor/plans_archived/stage1_forecastpipeline_completed.md`](../.cursor/plans_archived/stage1_forecastpipeline_completed.md)
