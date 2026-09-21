# Stage 3 — Speed, memory, and timing caveats

**Card:** NVIDIA A100-SXM4-40GB
**n:** 30 inits × 40 steps (10-day rollout), confirm logs `confirm-*-n30`, SHA `04cc501`
**Timer:** mean of `timing.full_rollout_ms / 40` (CUDA Events around `run_rollout`)
**VRAM:** `torch.cuda.max_memory_allocated` / `max_memory_reserved`

This note is the working record for WP6. n=6 screen RMSE stays out of any public table.

---

## Skill overlay (vs HRES-T0, plus WB2 n=730)

On n=30 skill vs HRES-T0, `tf32-matmul` and `bf16-amp-backbone` overlay the fp32 baseline on all 8 headline variables (max |δ| vs this n=30 fp32 vs-truth: 0.13% and 0.78%). Whole-model BF16 (`bf16-amp-full`, `bf16-weights`) sits well above that cluster (and WB2) from lead 1.

Skill overlay is not bitwise identity. Backbone Q2 vs the saved fp32 archive is already non-zero (z500 0.984 / 9.57 / 17.6 at day 1 / 5 / 10). That does not show on the vs-truth figure.

---

## Speed and memory (n=30 confirm)

Seconds/step = mean over 30 inits of CUDA `full_rollout_ms / 40`. That interval is **forward + per-step CPU offload**. Three warm-up forwards are discarded before the timed 40-step run. Finite check and FP32 cast run *after* the event pair (`cast_to_fp32=False`, then `finalize_rollout_predictions`).

| Variant | s/step | 40-step | vs fp32-baseline | Alloc | Reserved | Use? |
|---|---:|---:|---:|---:|---:|---|
| `fp32-baseline` (`confirm-fp32-baseline-n30`) | 6.24 | 249.6 s | 1.0× | 27.11 GiB | 37.57 GiB | baseline |
| `tf32-matmul` | 2.05 | 82.1 s | 3.0× | 27.11 GiB | 37.57 GiB | yes — fastest accurate row |
| `bf16-amp-backbone` | 2.09 | 83.5 s | 3.0× | 27.11 GiB | 36.02 GiB | yes — published AMP scope; not a VRAM win |
| `bf16-amp-full` | 1.37 | 54.8 s | 4.6× | 22.15 GiB | 31.18 GiB | no — RMSE |
| `bf16-weights` | 1.11 | 44.2 s | 5.6× | 18.18 GiB | 22.26 GiB | no — RMSE; citable #127 negative |
| `fp16-weights` / `fp16-weights-amp` | — | — | — | — | — | no forecast (`NonFinitePredictionError` on `msl`) |


| Variant | s/step | 40-step rollout | vs fp32-baseline | Max Memory Allocated |Max Memory Reserved|
|---|---:|---:|---:|---:|---:|
| `fp32-baseline`| 6.24 | 249.6 s | 1.0× | 27.11 GiB | 37.57 GiB |
| `tf32-matmul` | 2.05 | 82.1 s | 3.0× | 27.11 GiB | 37.57 GiB |
| `bf16-amp-backbone` | 2.09 | 83.5 s | 3.0× | 27.11 GiB | 36.02 GiB |
| `bf16-amp-full` | 1.37 | 54.8 s | 4.6× | 22.15 GiB | 31.18 GiB |
| `bf16-weights` | 1.11 | 44.2 s | 5.6× | 18.18 GiB | 22.26 GiB |
| `fp16-weights` / `fp16-weights-amp` | — | — | no forecast (`NonFinitePredictionError` on `msl`)  | — | — | |

Same rows, in plain English. `-amp` keeps the downloaded 32-bit weights and uses 16-bit copies inside selected ops. `-weights` converts the stored weights themselves.

| Variant | What we change |
|---|---|
| `fp32-baseline` | Nothing we set. Weights and matrix multiplies stay FP32 |
| `tf32-matmul` | Still 32-bit storage. Turns TF32 on for matrix multiplies (`float32_matmul_precision='high'`). Patch-embed conv TF32 was already on in the baseline. |
| `bf16-amp-backbone` | Weights stay 32-bit. Only the backbone runs in BF16 (`Aurora(autocast=True)`). Encoder and decoder stay 32-bit. |
| `bf16-amp-full` | Weights stay 32-bit. The whole forward pass is wrapped in BF16 AMP (`torch.autocast`). |
| `bf16-weights` | Convert the stored weights to BF16 and run with those. No AMP; arithmetic follows the resident 16-bit weights. |
| `fp16-weights` | Convert the stored weights to IEEE FP16 (`model.half()`). Mean sea-level pressure overflows to infinity. |
| `fp16-weights-amp` | `model.half()`, then FP16 AMP on the forward pass (starter-kit recipe). Same overflow on pressure. |

TF32 is a pure speed win: allocated and reserved match fp32 to the byte. Backbone-only AMP does not cut allocated VRAM. The only real VRAM cuts are the accuracy rejects. Usable 16-bit still peaks at 27.11 GiB, so the B>1 gate stays closed.

`full_rollout_ms` and `full_rollout_wall_s` agree to four figures on these runs. Mean of the 40 console `step_wall` values is the same loop chopped at `on_step`; it is **not** kernel-only time. `timing.per_step_ms` is 40× `None`.

---

## Caveats (timing)

### 1. Offload is inside the timer

`run_rollout` does `pred.to("cpu")` (device-to-host, D2H) **before** `on_step`. Both `full_rollout_ms` and `step_wall` include that copy. Switching the published number to mean(`step_wall`) would not remove offload.

That is the pipeline this repo actually runs: a 40-step list cannot stay on the A100. Stage 3 left offload in the CUDA pair on purpose so it would match Stage 2 (forward + offload). Host finite-check and FP32 cast were moved *out* of the pair.

### 2. D2H width is not equal across variants

The timed path does **not** cast to FP32 before offload. Native decoder output crosses the bus.

| Variant | Decoder output (DVO) | D2H width vs fp32 |
|---|---|---|
| `fp32-baseline`, `tf32-matmul` | fp32 | same |
| `bf16-amp-backbone` | fp32 (encoder/decoder stay fp32) | same |
| `bf16-amp-full` | bf16 | ~½ weather bytes |
| `bf16-weights` | bf16 | ~½ weather bytes |

So the ~3× rows are **not** skewed by transfer size. The cheaper D2H only helps the two RMSE rejects, and it makes them look slightly *faster*.

### 3. How large is that gift?

One lead is ~300 MiB of fp32 weather (~150 MiB bf16). A blocking D2H of that is ~10–20 ms, against 1.1–6.24 s of compute. Halving it is a ~10 ms/step gift, ~0.3–0.8% of reported s/step. It does not move 2.05 vs 1.11, and it does not change the reject.

Host RSS after the native-width list agrees with the dtype split (tf32/backbone ~13–14 GiB; amp-full/weights ~10.3 GiB). RSS is not the VRAM number.

### 4. Equalizing D2H would be a different protocol

Cast to FP32 **on the GPU** before `.to("cpu")` would equalize copy bytes. It would also add a bf16→fp32 kernel only on 16-bit-output rows (a different, smaller bias). Either protocol is fine if named. Current numbers are **native-width pipeline time**, not equalized-D2H kernel time. A true matmul-only figure needs CUDA Events around each `rollout` yield *before* the copy.

### 5. Do not mix timers without naming them

Confirm `fp32-baseline` n=30 (`confirm-fp32-baseline-n30`, SHA `04cc501`) uses the same CUDA pair as the other confirm rows: **6.24 s/step**, 249.6 s / 40-step, 27.11 GiB allocated, 37.57 GiB reserved. Q2 vs the Stage 2 persist headline archive is RMSE **0** on every scored row (9 600). Speedups in the table above are vs this refresh, not vs persist 6.31 s/step. Do not mix the persist wall time into the confirm table.

### 6. Public wording

- Quote n=30 and the A100-SXM4-40GB on every number.
- n=6 screen RMSE never leaves the repo.
- Put the D2H-width caveat on `bf16-amp-full` / `bf16-weights` only, not on the tf32 vs fp32 headline.
- FP16: scoped claim only — this `.half()` (+ optional AMP) recipe on `aurora-finetuned` overflowed global `msl`. Do not claim Microsoft’s default path is quietly wrong.
