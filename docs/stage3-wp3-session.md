# Stage 3 WP3 session script (draft — review before booking)

Laptop proofs for WP2 are done. This is the **first GPU booking** of the mixed-precision slice: a 10-step n=6 screen on **A100-SXM4-40GB**. Do not book until this page has been reviewed and `make check` is green on `stage3_mixed_precision_study`.

**`--max-steps` is implemented** on `scripts/run_fidelity.py`. Do not edit the frozen screen TOML. `--cpu-dry-run` is not a substitute for anything on this page.

---

## Session definition of done (fill before launch)

```
Date / timebox end: _______________  (hard stop — terminate even if DoD incomplete)
SKU booked: NVIDIA A100-SXM4-40GB
Image: GPU Base 22.04
ISA expected: x86_64
PLATFORM: linux/amd64
Hourly rate verified at booking: $_____ / hr
Billing alarm: wall-clock timer + terminate at timebox (book 3 hours)

This session DoD:
  [ ] SSH as ubuntu; uname -m = x86_64; nvidia-smi shows A100-SXM4-40GB
  [ ] NFS attached: splice + headline archive (same region as the GPU)
  [ ] Clone stage3_mixed_precision_study; PLATFORM=linux/amd64 make docker-build-gpu
      (or uv sync --extra forecast --frozen on the box)
  [ ] WP2.3 archive re-score (step 2 below) — STOP the session if it aborts
  [ ] --max-steps 10 passed on every variant command (TOML stays 40)
  [ ] Three discarded warm-up forwards (harness already does this)
  [ ] Six precision variants, n=6, 10 steps, in the order below
  [ ] One run log per variant per starting date; artefacts on NFS / scp
  Teardown
  [ ] scp from the laptop
  [ ] Instance TERMINATED (not Stop); console gone; billing stopped
  [ ] NFS stays attached only if the GiB-month $ is still written down

Out of scope this booking:
  - n=30 confirm (WP5)
  - --max-steps on the 6-date TOML (do not edit the TOML)
  - Batching, torch.compile, 8-bit, per-module precision
  - Re-persisting the Stage 2 archive
  - Publishing any n=6 number
```

---

## NFS paths (do not copy off the volume)

| What | Path |
|---|---|
| HRES-T0 splice | `/home/ubuntu/xerxes-nfs/splice_hres_t0_2022_full.zarr` |
| Headline archive | `/home/ubuntu/xerxes-nfs/aurora-baselines/wp5b-baselines-n30-headline/` |
| Stage 2 skill tables | `…/wp5b-baselines-n30-headline/rmse_by_init.csv` (persist vs HRES-T0) |
| Screen TOML | `configs/hres_t0_2022_fidelity_screen_rollout.toml` (frozen) |

Pairing is persist `init_id` by `init_time` (1, 6, 15, 16, 22, 24), not screen ordinal 1..6.

---

## Order of operations

### 1. Write the session environment file

Same as Stage 2: `uname -m`, `nvidia-smi`, git SHA, `torch` + CUDA, hygiene readback. The fidelity harness writes `session.json` per variant. Also copy this DoD into `~/session-artifacts/`.

### 2. WP2.3 — re-score archive leads (FIRST compute; 10 min)

The WP1.5 scorer now accumulates in FP64. Confirm that does not move the Stage 2 skill numbers on **real** archive vs HRES-T0 pairs. Self-vs-self is the WP1.5 blind spot — do not do it.

```bash
uv run --extra forecast python scripts/rescore_baseline_leads.py \
    --baseline-dir /home/ubuntu/xerxes-nfs/aurora-baselines/wp5b-baselines-n30-headline \
    --splice-path /home/ubuntu/xerxes-nfs/splice_hres_t0_2022_full.zarr \
    --init-id 1 \
    --leads 24 120 240 \
    --output-dir /home/ubuntu/xerxes-nfs/aurora-fidelity/wp2-archive-rescore \
    --tag wp2-3-rescore
```

- Init 1 is `2022-01-01T12`. Leads 24 / 120 / 240 h are day 1 / 5 / 10 (`FLOOR_LEAD_HOURS`).
- Compares new `MSE` (FP64 accumulate) to Stage 2 `rmse_by_init.csv`.
- Exit 1 if any row's relative change exceeds `1e-4` (session abort heuristic, **not** the 5% skill band).
- **If it aborts: stop. Do not spend the rest of the booking on variants.**

### 3. Warm-up

Three discarded 1-step forwards, already in `scripts/run_fidelity.py` `_warmup`. Do not skip.

### 4. Precision screen (`--max-steps 10`)

`--max-steps` truncates the frozen 40-step screen TOML at runtime. Do not edit the TOML.

```bash
# Truncate at 10 steps; do not edit the screen TOML.
for variant in tf32-matmul bf16-amp-backbone bf16-amp-full \
               bf16-weights fp16-weights fp16-weights-amp; do
  uv run --extra forecast python scripts/run_fidelity.py \
      --variant "$variant" \
      --baseline-dir /home/ubuntu/xerxes-nfs/aurora-baselines/wp5b-baselines-n30-headline \
      --rollout-config configs/hres_t0_2022_fidelity_screen_rollout.toml \
      --output-dir /home/ubuntu/xerxes-nfs/aurora-fidelity \
      --tag "screen-${variant}-s10" \
      --max-steps 10
done
```

Order is cheapest-and-safest first. A session that dies early still returns something. Cut `fp16-weights*` first if budget bites; **never cut `bf16-weights`**.

Pairing at 10 steps covers leads 6–60 h only. Unmatched archive leads (66–240 h) are unused — not a pairing bug.

### 5. Artefacts to keep

Per variant: run JSON (DVO, observed formats, inverted-zero, timings, VRAM), `vs_baseline/` and `vs_truth/` RMSE tables. Screen n=6 numbers **stay in the repo / NFS**. They do not go in the README, email, or #127.

### 6. Copy off, terminate, confirm billing stopped

`scp` from the laptop. Terminate, not stop. NFS still bills if attached.

---

## Review notes (WP2 close)

- WP2.3 is scripted (`scripts/rescore_baseline_leads.py`). It is **not** executed on the laptop.
- `--max-steps` is on `scripts/run_fidelity.py`. The loop in §4 is the booking command. Do not edit the screen TOML.
- Skill gate remains the existing 5% band. The 1e-4 abort in WP2.3 is only “did the scorer rewrite move Stage 2 numbers”.

---

## Appendix A — WP2.1 laptop dry-run (no skill)

Command:

```bash
uv run python scripts/dry_run_precision.py --output-dir outputs/wp2-precision-dry-run
```

Untrained `AuroraSmallPretrained`, synthetic 32×64, one init, two steps. Same precision helpers as the `aurora-finetuned` factories. `check_finite=False` (untrained NaNs).

| Variant | Outcome | DVO | Notes |
|---|---|---|---|
| `tf32-matmul` | ran | PASS | Flag set to `high`. TF32 kernels are CUDA Ampere; CPU arithmetic stays host FP32. Recorded in run-log warnings. |
| `bf16-amp-backbone` | ran | PASS | Wrapper tensors stay FP32; backbone LayerNorm is bf16. DVO counts that LayerNorm toward `scope`. |
| `bf16-amp-full` | refused | `REFUSED — whole-forward autocast requires CUDA; …` | Must not silently run FP32. |
| `bf16-weights` | ran | PASS | Resident weights bf16; weather-only batch cast. |
| `fp16-weights` | ran | PASS | Resident weights fp16; weather-only batch cast. |
| `fp16-weights-amp` | refused | `REFUSED — whole-forward autocast requires CUDA; …` | Must not silently run FP32. |

`--cpu-dry-run` was not used (still self-vs-self `fp32-baseline`, never calls `model_factory`).

---

## Appendix B — WP2.2 scorer gap vs `math.fsum`

Grid: 721×1440, truth = 280, forecast = 280 × (1 + 1e-4) relative, latitude-weighted MSE. Old FP32 scorer reconstructed in `tests/test_eval_metrics.py` only. Production `MSE` stays FP64 accumulate. Reference: `math.fsum` (not `Decimal`).

| Accumulator | MSE | Relative gap vs `math.fsum` |
|---|---|---|
| FP32 (reconstructed pre-WP1 path) | 7.848468376e-4 | 1.335e-6 |
| FP64 (production `MSE`) | 7.848470705e-4 | 1.038e-6 |
| `math.fsum` | 7.848478854e-4 | 0 (reference) |

Laptop, 2026-09-20. Constant 280 field, 1e-4 relative offset, 721×1440, latitude-weighted. The close-variant-sized MSE is ~7.85e-4; both gaps are ~1e-6 of that quantity — not a meaningful fraction of the signal. **FP64 accumulation is cheap insurance at this size, not a rescue of an unusable scorer.** Still required: Stage 2's FP32-against-itself floor is exactly zero, so it could not have detected accumulation rounding. Keep the production scorer on FP64.
