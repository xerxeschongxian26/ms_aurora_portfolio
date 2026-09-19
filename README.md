# How much precision does Aurora actually need?
<sub>_This is an independent study, not affiliated with or endorsed by Microsoft or the author's employer._</sub>

> **_Insert example key result here_**  
> e.g. bf16 autocast on a single A100-40GB: `[X]`× faster, `[Y]`% less peak VRAM, forecast RMSE within `[Z]`% of full precision to day 10.

This repository presents an independent study of mixed-precision inference for [Microsoft Aurora](https://github.com/microsoft/aurora) (0.25° fine-tuned checkpoint).  
To my knowledge, no published characterisation of this trade-off exists for Aurora.

<!-- EYE-CATTCHING - VISUALS 10-day rollout animation -->
<p align="center">
  <img src="/Users/xerxeschongxian26/Desktop/Turning Pro/ms_aurora_portfolio/docs/images/hres_t0_forecast_2t.png" alt="10-day 2m temperature forecast" width="720"/>
  <br><em>10-day global 2m temperature forecast from <insert time here> initalisation point, produced by this pipeline.</em>
</p>

---

## Results

**Hardware:** Single NVIDIA A100-SXM4-40GB  
**Software:** Microsoft-Aurora `1.8.0` · PyTorch `2.12.1+cu130` · CUDA `runtime 13.0`

| Variant | What is cast | s / step | Peak VRAM (GiB) | Δ RMSE vs baseline, z500 | Δ RMSE vs baseline, t850 | Δ RMSE vs baseline, 2t | n |
|---|---|---|---|---|---|---|---|
| `fp32` (baseline) | cuDNN TF32 by default, matmul fp32 | `[X]` | `[X]` | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 30 |
| `tf32` | matmul + cuDNN | `[X]` | `[X]` | `[X]` | `[X]` | `[X]` | 30 |
| `bf16-autocast` | backbone only (encoder/decoder stay fp32) | `[X]` | `[X]` | `[X]` | `[X]` | `[X]` | 30 |

### Key findings

- `[Finding 1]`
- `[Finding 2]`

### Figures

<_insert figure of overlapping RMSE plots of variants vs baseline_>

---

## Can we trust the baseline?

Before measuring precision trade-offs, the codebase was validated against the benchmark performance reported by [WeatherBench2](https://weatherbench2.readthedocs.io/) (Rasp et al., 2024).

### Matched Out-Sample Performance

<p align="center">
  <img src="docs/images/baseline_30inits_rmse_vs_lead_time.png" alt="Headline RMSE vs lead: this pipeline n=30 vs WeatherBench2 n=730" width="720"/>
</p>

<em>Headline RMSE vs lead time. WeatherBench2 Aurora vs HRES-T0 (n=730) overlaid with this pipeline’s Q1 skill run (n=30), 2022.</em>

- **8 headline variables** (z500, t850, 2t, 10u, msl, u500, t500, q500), n=30 inits from 2022 (held out from training), within 5% of the WB2 n=730 reference at all lead times up to and including day 10.

### Bitwise Reproducibility
- **Baseline floor:** fp32 vs fp32 RMSE = 0 across 1,920 comparisons. The setup is bitwise reproducible on this hardware. <span style="color:red">EXPAND ON THIS</span>

## Method

| | Detail |
|---|---|
| **Model** | `microsoft/aurora` 0.25° fine-tuned ( Checkpoint [`0be7e57`](https://huggingface.co/microsoft/aurora/commit/0be7e57)) |
| **Data** | HRES-T0 analysis, 2022 out-of-sample periodm, 00/12 UTC initialisation points only <span style="color:red">DESCRIBE HOLES?</span> |
| **Metric** | Latitude-weighted RMSE  <span style="color:red">footnote, refactored WB2 formula</span> |
| **What is cast** | `bf16-autocast`: Aurora's `autocast=True` (AMP bf16, backbone only). `tf32`: `float32_matmul_precision="high"` + `cudnn.allow_tf32=True`. |
| **Hardware** | Single NVIDIA A100-SXM4-40GB, PyTorch `2.12.1+cu130`, CUDA `runtime 13.0` |
| **Init sets** | n=6 screen (1/quarter + 2 high-gradient cases), n=30 confirm (year-spread) <span style="color:red">expand? this is confusing</span>|

- <span style="color:red">Describe selected init points further</span> 
- All timings recorded using CUDA Events with 3-step warm-up
- Memory recorded via `torch.cuda.max_memory_allocated`  
---

## Limitations

- **n=30**, not the supplementary information's full 730-init out-sample testing protocol
- Single GPU - A100-SXM4-40GG
- RMSE on _headline variables_ only — no ACC, no extreme-event metrics
- Deterministic model only (not Aurora 1.5 ENS)
---

## Roadmap

| Stage | Focus | Status |
|---|---|---|
| 0 | Environment, Docker, dependencies | ✅ Done |
| 1 | Forecast pipeline — end-to-end inference | ✅ Done |
| 2 | Evaluation harness — Q1 skill + Q2 fidelity wiring | ✅ Done |
| 3 | **Inference optimisation** — mixed precision, then batching only if peak VRAM drops; optional PTQ | 🔧 In progress |
| 4 | Report, figures, optional extended evaluation | 📝 Planned |
| 5 | **Inference serving** — one init in, forecast out | 📝 Planned |

Stage 3 is gated, not a checklist: (1) `tf32` and `bf16-autocast` vs the saved fp32 maps; (2) `B>1` **only if** mixed precision actually lowers peak VRAM on this A100 40GB (fp32 already OOMs at `B=2`); (3) hand-rolled post-training quantisation is an **appendix**, not a gate. Then Stage 4 write-up and Stage 5 serving with the cheapest variant that stayed inside the RMSE budget.

## Contributions to upstream

[PR #196](https://github.com/microsoft/aurora/pull/196): **Fix silent metadata–tensor shape mismatch in `Batch`** — merged into `microsoft/aurora`. Added post-init validation that catches mismatched time and pressure-level dimensions before they propagate silently through the model. Discovered via Issue [#188](https://github.com/microsoft/aurora/issues/188) during this work.

## Acknowledgements and Sources

**Aurora** (Bodnar et al., 2024)
- [Code](https://github.com/microsoft/aurora) · [Weights](https://huggingface.co/microsoft/aurora) · [Docs](https://microsoft.github.io/aurora/intro.html)
- [Paper](https://www.nature.com/articles/s41586-025-09005-y)
- [Supplementary Information](https://www.nature.com/articles/s41586-025-09005-y#Sec29)
- [Aurora 1.5, an update](https://www.microsoft.com/en-us/research/publication/aurora-1-5-fine-tuning-a-foundation-model-for-medium-range-ensemble-weather-prediction/)

**WeatherBench 2** (Rasp et al., 2024)
- [Paper](https://doi.org/10.1029/2023MS004019)
- [Docs](https://weatherbench2.readthedocs.io/)
- [Aurora vs HRES-T0, 2022](https://storage.googleapis.com/weatherbench2/benchmark_results/aurora_vs_hres_t0_1440x721_2022.nc)

Links last accessed 18 September 2026.

## License

MIT — see [LICENSE](LICENSE).

