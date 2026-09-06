# GPU memory scaling (2026-09-05)

Two Lambda sessions. Same image and process
- `aurora-inference:gpu-linux-amd64`, `PLATFORM=linux/amd64`
- `scripts/real_forecast.py`
- `aurora-finetuned` i.e. 0.25° grid, $721 \times 1440$
- single default initialisation point in `scripts/real_forecast.py`: `2022-06-15T12`
- Batch Dimension, B = 1
    - i.e. Single initialisation point
    - inputs are surf `2t=(1, 2, 721, 1440)`
    - atmos `t=(1, 2, 13, 721, 1440)`
- Batch Dimension, B > 1
    - i.e. Multiple initialisation points
    - In practice, each individual initialisation point and its associated forecast/rollout is independent from the rest
    - For investigating VRAM usage, forecast skill is not important hence a multi-batch is created using `_repeat_batch`
        - real copies of that one init, not unique initialisation points

| Machine | VRAM Available | Runs (folder names) |
|---|---|---|
| A100 40 GB SXM4 | 40960 MiB | `b1-s1-cold-cache`, `b1-s1-warm-cache`, `b1-s2`, `b1-s4`, `b2-s1` (OOM) |
| H100 80 GB SXM5 | 81920 MiB | `b1-s1-cold-cache-h100`, `b1-s1-warm-cache-h100`, `b2-s{1,2,4}-h100`, `b3-s1-h100`, `b4-s1-h100` (OOM) |

SXM is the GPU's board/socket (not the chip): SXM4 is the A100 generation, SXM5 the H100 generation — it does not change allocated VRAM (B=1 peak is ~26.3 GiB on both); Hopper only shortens wall time.

Folders without `h100` suffix are based on the A100 machine. Run artifacts are found in `outputs/<folder>/` (gitignored).
- Focus on the **directory** names, some logs still print the original `--tag` (e.g. cold H100 was tagged `b1-s1-tbd`)

## Meric Loggers

Every log entry pertaining to memory is created by `_log_mem` in `scripts/real_forecast.py`.

| Log field | Script helper | Underlying call |
|---|---|---|
| peak RSS | `peak_rss_bytes()` (`src/aurora_inference/logging.py`) | `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` |
| VRAM now | `_vram_bytes()` | `torch.cuda.memory_allocated` |
| VRAM peak | `_vram_bytes()` | `torch.cuda.max_memory_allocated` |

Peak RSS is a high-water mark since process start. It never decreases.

Before each segment of the script, `_reset_vram_peak` calls `torch.cuda.synchronize` then `torch.cuda.reset_peak_memory_stats`.
VRAM peak in the next `_log_mem` is since that reset, not process start.
Both VRAM helpers also `synchronize` before they read.

| Segment label | Code snippet timed / measured |
|---|---|
| `after batch.to` | `batch.to(_DEVICE)` |
| `after model load` | `load_model` (wall time measured using `time.perf_counter`) |
| `after rollout` | `run_rollout` (wall time measured using `time.perf_counter`) |
| `after plots` | `plot_batch_2t` loop |

Hopper generation (H100) is a faster chip than the Ampere generation (A100)

## A100 40 GB — numbers

| Seam | b1-s1 cold | b1-s1 warm | b1-s2 | b1-s4 | b2-s1 |
|---|---:|---:|---:|---:|---:|
| input `2t` | (1, 2, 721, 1440) | (1, 2, 721, 1440) | (1, 2, 721, 1440) | (1, 2, 721, 1440) | (2, 2, 721, 1440) |
| `after batch.to`  VRAM now| 558.85 | 558.85 | 558.85 | 558.85 | 1105.89 |
| model load wall | 40.17 s | 19.11 s | 18.19 s | 18.60 s | 46.12 s |
| `after model load` VRAM now | 5369.17 | 5369.17 | 5369.17 | 5369.17 | 5916.29 |
| `after model load` peak RSS | 10856 | 10738 | 10713 | 10709 | 10836 |
| rollout wall | 6.28 s | 6.29 s | 12.35 s | 24.50 s | OOM |
| `after rollout` VRAM now | 5851.69 | 5851.69 | 6137.72 | 6707.94 | — |
| `after rollout` VRAM peak | 26926.73 | 26926.73 | 27756.52 | 28327.99 | — |

B=1 per-step wall (warm): 6.29 s, 6.18 s, 6.13 s.
`b2-s1` log ends after model load; no PNG. CUDA OOM in `run_rollout`. Input VRAM is ~2× B=1 (1106 vs 559 MiB), so the copies landed.

## H100 80 GB — numbers

| Seam | b1-s1 cold | b1-s1 warm | b2-s1 | b2-s2 | b2-s4 | b3-s1 | b4-s1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| input `2t` | (1, …) | (1, …) | (2, …) | (2, …) | (2, …) | (3, …) | (4, …) |
| `after batch.to` VRAM now | 558.85 | 558.85 | 1105.89 | 1105.89 | 1105.89 | 1652.52 | 2199.89 |
| model load wall | 36.92 s | 12.42 s | 12.73 s | 12.45 s | 12.39 s | 12.37 s | 12.28 s |
| `after model load` VRAM now | 5369.17 | 5369.17 | 5916.29 | 5916.29 | 5916.29 | 6462.92 | 7010.29 |
| rollout wall | 2.67 s | 2.63 s | 5.09 s | 10.05 s | 20.02 s | 7.64 s | OOM |
| `after rollout` VRAM now | 5875.57 | 5875.57 | 6697.77 | 7255.84 | 8372.20 | 7514.89 | — |
| `after rollout` VRAM peak | 26950.41 | 26950.41 | 50374.58 | 52027.12 | 53143.55 | 73035.84 | — |

B=1 per-step wall (warm): 2.63 s. B=2 per-step: 5.09 s, 5.03 s, 5.01 s.
`b4-s1-h100` ends after model load; no PNG. OOM in `run_rollout`.

## Observations

- **Allocated VRAM is almost SKU-independent.** B=1 s=1 peak is 26926.73 MiB on A100 and 26950.41 MiB on H100. Weights after load are 5369.17 MiB on every successful B=1 run. The 40 GB / 80 GB figures are board size, not this meter.

- **Steps scale time, not the activation peak.** On A100, B=1 peak stays in a ~26.3–27.7 GiB band (26927 / 27757 / 28328). On H100, B=2 peak stays in a ~49.2–51.9 GiB band (50375 / 52027 / 53144). Extra steps keep extra predictions on GPU (`run_rollout` does `list(rollout(...))` and does not `.to("cpu")`); they do not stack another `forward`. See below for 40-step pressure on a 40 GB card.

- **`B` scales the forward peak almost linearly.** H100 s=1 peaks: 26.32 / 49.19 / 71.32 GiB for B=1/2/3. Step from B=1→2 is +22.9 GiB; B=2→3 is +22.1 GiB. That extra is activations / attention workspace, not a second copy of the 5 GiB weights.

- **Resident after load grows only by the extra input copies.** +547 MiB per added `B` (5369 → 5916 → 6463 → 7010). `after batch.to` is 559 / 1106 / 1653 / 2200 MiB.

- **`VRAM now` after rollout is leftover tensors.** Activations are freed. Now = weights + input + kept preds. It grows with both `B` and steps (B=2 s=1/2/4: 6698 / 7256 / 8372 MiB).

- **SKU fit.** B=1 (~27 GiB allocated) fits A100 40 GB. B=2 (~50 GiB) OOMs A100 and fits H100 80 GB. B=3 (~71 GiB allocated) fits H100 with little headroom. B=4 would be ~93 GiB if the +22 GiB step continues — OOM on 80 GB, as observed.

- **Hopper is faster, not smaller.** Warm B=1 s=1 rollout is 6.29 s on A100 vs 2.63 s on H100 (~2.4×). Warm B=2 s=1 is 5.09 s (about 1.9× a B=1 H100 step, not 2.0×). Do not mix SKUs when quoting seconds.

- **Cold cache only ruins load wall.** A100 load 40.17 s → 19.11 s warm; H100 36.92 s → 12.42 s. Rollout VRAM peak is identical cold vs warm on the same SKU+B+steps.

- **Peak RSS ~11 GiB at load is two host copies of the weights, not RAM + VRAM.** `load_checkpoint` deserialises the `.ckpt` into CPU tensors while `Aurora()` parameters also live on the CPU, so host RAM briefly holds ~5 GiB + ~5 GiB. `model.to("cuda")` is after that spike: it writes weights to VRAM and does **not** add 5 GiB to RSS. After the CPU copies are dropped, live host RAM can fall; the log stays at ~10.7–10.9 GiB because `ru_maxrss` is a high-water mark. GPU weights (~5 GiB in `after model load` VRAM) are not part of RSS.

## 40-step B=1 on A100 40 GB

Each `forward` is still T=2 in, T=1 out. Op shapes do not grow with lead time, so wall time from 1/2/4 steps (~6.2 s/step → ~4.1 min for 40 steps) is a good **compute floor**.

The leftover list is the risk, not the op shapes. `run_rollout` does `list(rollout(...))` and keeps every prediction on CUDA. `VRAM now` after rollout grows ~286 MiB per extra step (5852 → 6138 → 6708 MiB). By step 40 that is about `5852 + 39 × 286` ≈ **17 GiB** resident and peak ≈ `26927 + 39 × 286` ≈ **37 GiB allocated** on a 40 GiB board.

Allocated tensors staying the same size does not mean the allocator stays happy. Fragmentation and a nearly full board can stretch a step or OOM. That would look like later steps got slower, even though the model is unchanged. `nvidia-smi` sits above allocated, so a 40-step run can still fail when 37 GiB < 40 GiB.

If `run_rollout` iterates the official generator and does `pred.to("cpu")` each step (new `Batch`; do not mutate the yielded tensors in place), resident stays near s=1 and the linear time estimate is solid. Do not change `aurora.rollout`; it still needs the GPU `pred` for the next-step `torch.cat`. If all 40 preds stay on CUDA, treat 4.1 minutes as the floor and the last steps as the risk.

## Takeaway

One `forward` sets the VRAM peak. Extra **steps** cost wall time. Extra **B** costs VRAM (~23 GiB allocated per added batch on this grid). A100 40 GB is a B=1 card for full 0.25° `aurora-finetuned`. H100 80 GB holds B=2 and B=3; B=4 does not. For many inits on 40 GB, run sequential B=1 (or one init per GPU), do not stack `B`. For a 40-step rollout on this SKU, offload each pred to CPU or expect ~37 GiB allocated plus allocator pressure.
