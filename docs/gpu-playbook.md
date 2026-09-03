# GPU playbook (Lambda Labs)

Stage 1 GPU work is **closed** (2026-08-16). This file is the runbook to **reproduce** that session — HRES-T0 → `Aurora` 0.25° FT → global `2t` maps — and for any later paid GPU hour (Stage 2 eval). Debug the data path on CPU first; do not debug `HresT0Source` on the clock.

**Standing rule:** every paid session starts with a written definition of done and ends with a **full terminate** (not stop). Forgotten instances destroy the budget.

**Verified Stage 1 session (2026-08-16):** Lambda **1× A100 40 GB SXM4**, Virginia, **GPU Base 22.04**, `uname -m` = `x86_64`, `PLATFORM=linux/amd64`. First boot was plain Ubuntu 22.04 (no `nvidia-smi`) → **Terminate**, relaunch GPU Base. Entrypoint: `scripts/real_forecast.py` (image `CMD`). Four `2t` maps, eyeballed plausible. `scp` from the laptop. Instance **terminated**; console empty.

A Stage 0 **CPU-image** lifecycle rehearsal (2026-07-21, 1× A10) is in the appendix. Do not use that path for a real forecast.

---

## 0. Before you click Launch

### Session definition of done (fill this in *before* launch)

Copy, edit, keep visible until teardown:

```
Date / timebox end: _______________  (hard stop — terminate even if DoD incomplete)
SKU booked: _______________________  (prefer A100 40GB+ / H100)
Image: GPU Base 22.04
ISA expected: x86_64 | aarch64     (fill from console, confirm with uname -m)
PLATFORM: linux/amd64 | linux/arm64
Hourly rate verified at booking: $_____ / hr
Billing alarm / spend guard: wall-clock timer + terminate at timebox

This session DoD (tick as you go):
  [ ] SSH as ubuntu; uname -m and nvidia-smi match the SKU
  [ ] Docker group active; GPU visible in container
      docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi
  [ ] docker manifest inspect covers booked PLATFORM for 12.6.3-runtime-ubuntu22.04
  [ ] Clone repo; PLATFORM=… make docker-build-gpu succeeds
  [ ] Trivial CUDA forward inside the gpu image (torch.cuda.is_available)
  [ ] HF cache mounted; aurora-0.25-finetuned.ckpt loads
  [ ] Forecast: scripts/real_forecast.py --steps 4  (HRES-T0 2022-06-15T12, naive datetime)
  [ ] Eyeball: 2t maps coherent, continents upright, weather evolves (not skill)
  Teardown
  [ ] scp PNGs + logs from the laptop (not from the instance)
  [ ] Instance TERMINATED (not Stop); console gone; billing stopped
  [ ] If NFS was attached: either keep it only with a written GiB-month $ (§0), or DELETE the filesystem (terminate does not stop NFS billing)

Out of scope unless this session is Stage 2:
  - RMSE / ACC / metrics
  - Regional cropping, optimization, folding zarr/gcsfs into core
  - Debugging HresT0Source on the GPU clock (abort and terminate)
  - Leaving the instance stopped overnight
```

### Billing / spend control

- Re-check the **current hourly rate** and SKU availability at booking (both move).
- Lambda may **not** expose a first-class billing alarm — use a wall-clock timebox and terminate when it ends.
- Stage 1 used **1× A100 40 GB**. Prefer GH200 when it is in stock; otherwise x86 A100/H100. Do not book a cheap A10 for a full-grid FT forecast.

### Persistent filesystem (Lambda NFS)

Lambda **persistent filesystems** are NFS volumes: they mount on the VM, **survive instance terminate**, and bill **while the filesystem exists** — including when no instance is attached.

**Short GPU session (Stage 1-style, hours not days):** **do not** attach one. Use the instance local SSD; `scp` artifacts home before terminate.

**Reuse across Stage 2/3 sessions (HRES-T0 splice, FP forecasts):** attach **at launch**, **same region** as the GPU, only after the monthly $ is written below. Delete the filesystem at phase closeout — terminating the VM does **not** stop NFS billing.

#### Rate (confirm in the launch UI; it moves)

Lambda’s published example matches the rate shown at instance launch (2026-09):

> Filesystems are billed per GiB **used** per month in **one-hour** increments. For example, at **$0.20 per GiB per month**:
>
> - 1,000 GiB continuously for a full month (720 hours) → **$200.00**
> - 1,000 GiB continuously for a full day (24 hours) → **$6.67**

So: `cost ≈ (GiB used) × $0.20 × (hours / 720)`. You are not stuck with a calendar month if you **delete** the volume.

#### HRES-T0 size → NFS $ (uncompressed ballpark, $0.20/GiB-month)

Per 6 h timestamp at 0.25° (721×1440, float32): 9 Aurora fields (4 surf + 5 atmos × 13 levels) ≈ **0.267 GiB**. Unused WB2 vars (`10m_wind_speed`, `vertical_velocity`, precip, …) are **not** Aurora inputs — do not copy them onto NFS.

| What you persist | Size (order of mag.) | Full month | ~24 h |
|---|---|---|---|
| Dev splice: 10 overlapping 00/12 inits, 14.75 d (`2022-01-01T06`–`2022-01-16T00`), 9 Aurora vars, **60** times | ~16 GiB | **~$3** | negligible |
| 2022 + Jan 2023 tail, **9 Aurora vars only**, ~1500 times | ~400 GiB | **~$80** | ~$2.70 |
| Full HRES-T0 including unused vars | ~550 GiB | **$110** | ~$3.70 |

Predictions are **on top of** the analysis splice. Overlapping inits still produce **N × 40** distinct rollouts (Aurora output grid **720×1440**). Ten inits × 40 steps, full fields ≈ **~107 GiB** extra (~**+$21/month**); z500+t850 only ≈ **~3 GiB**.

#### Decision rule

- Dev / plumbing: laptop or the **~16 GiB** splice. NFS is optional (~$3/month) if you will remount it.
- Skill vs Fig H7 / many GPU days: NFS for the **Aurora-only year** (~$80/month), not the 18-variable store (~$110/month).
- One or two GPU days and no reuse: ephemeral SSD + subset download; **do not** leave 400–550 GiB sitting idle for a month.
- Record at booking: rate, GiB, monthly $ if left up. **Delete the filesystem** when Stage 2/3 stops — same discipline as terminating the instance.

---

## 1. Target

| Preference | SKU | ISA | `PLATFORM` | When |
|---|---|---|---|---|
| **Preferred** | Lambda **GH200** (H100, 96 GB) | `aarch64` | `linux/arm64` | When in stock |
| **Working (2026-08)** | A100 / H100 on Lambda | `x86_64` | `linux/amd64` | GH200 unavailable (Stage 1 case) |

Do **not** treat an x86 box as “the same as GH200.” Same Docker *recipe*, different host class. Confirm with `uname -m` **on the instance** (not on your Mac).

| Where you run `uname` | Typical result |
|---|---|
| Mac (Apple Silicon) | `uname` → `Darwin`; `uname -m` → `arm64` |
| Lambda A10 / A100 / H100 | `uname` → `Linux`; `uname -m` → `x86_64` |
| Lambda GH200 | `uname` → `Linux`; `uname -m` → `aarch64` |

The GPU image is `nvidia/cuda:12.6.3-runtime-ubuntu22.04`. Keep container CUDA in **12.4–12.8**. Torch CUDA-index pin (`cu124`/`cu126`) is still a `TODO(stage-2)` in the Dockerfile; WP6 used `uv sync --frozen --extra forecast` without an explicit index and CUDA was available.

---

## 2. Launch

1. Lambda Cloud console → create instance.
2. Pick region/SKU with stock; **write down $/hr**.
3. **Image (critical):** **GPU Base 22.04** (Ubuntu 22.04 family). Prefer **22.04** over 24.04 to stay aligned with `nvidia/cuda:*-ubuntu22.04`.
4. **Do not** pick a plain “Ubuntu 22.04 / 24.04” base. Lesson (Stage 1): plain Ubuntu still shows the GPU in `lspci`, but **`nvidia-smi` is missing**. Do **not** `apt install nvidia-utils-*`. **Terminate** and relaunch **GPU Base**.
5. Attach your SSH public key at create time.
6. Persistent filesystem: skip for short sessions; Stage 2/3 HRES-T0 splice only after the §0 NFS cost is written down (same region, attach at launch).
7. Wait until running; copy the public IP.
8. Start a local timer for the timebox.

---

## 3. SSH

```bash
ssh ubuntu@<INSTANCE_IP>
```

User **must** be `ubuntu`. Lambda injects keys into `ubuntu`'s `authorized_keys`; custom usernames fail.

If Cursor pops “application running on port 22”: that is the **SSH** port for this session, not Aurora or Docker. Dismiss and continue in the terminal.

---

## 4. First-boot

```bash
uname -m          # expect: x86_64 (A100/H100) or aarch64 (GH200)
nvidia-smi        # expect: the GPU you booked
# If nvidia-smi missing but lspci | grep -i nvidia shows the card:
#   wrong OS image → terminate and relaunch with GPU Base (§2). Do not apt-install drivers.

sudo usermod -aG docker "$USER"
```

Group membership does **not** apply to the current shell. Either:

```bash
exit
ssh ubuntu@<INSTANCE_IP>    # fresh login
# or, in the same session (often needed in Cursor remote terminals):
newgrp docker
```

Then:

```bash
groups            # must list docker
docker info       # no permission errors on the socket
```

**Lesson:** `usermod -aG docker` updates `/etc/group`, but Cursor/SSH sessions often keep the old group set → `permission denied … docker.sock`. `newgrp docker` (or a brand-new SSH login) fixes it. Prefer fixing the group so `make docker-build-gpu` does not need `sudo`.

---

## 5. Toolkit regression (GPU visible in Docker)

On a new SKU this is a **regression check**. If it fails, stop before building project images.

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi
```

Expect the booked GPU inside the container output.

```bash
docker manifest inspect nvidia/cuda:12.6.3-runtime-ubuntu22.04
```

Confirm a manifest exists for the platform you will use.

---

## 6. Clone, build, run (GPU image + HRES-T0 forecast)

### Match `PLATFORM` to the instance (mandatory)

| `uname -m` on instance | `PLATFORM` |
|---|---|
| `x86_64` | `linux/amd64` |
| `aarch64` | `linux/arm64` |

Makefile default is `linux/arm64` (GH200 / M1). On an A100, plain `make docker-build-gpu` builds the **wrong** arch and fails at `uv sync` with `exec format error`. Always pass `PLATFORM` explicitly on cloud boxes.

```bash
git clone https://github.com/xerxeschongxian26/ms_aurora_portfolio.git
cd ms_aurora_portfolio
# Until stage1 is merged to main:
#   git clone -b stage1 --single-branch https://github.com/xerxeschongxian26/ms_aurora_portfolio.git

export PLATFORM=linux/amd64   # on x86 Lambda; use linux/arm64 on GH200
make docker-build-gpu PLATFORM="$PLATFORM"
```

The **`gpu`** stage installs `uv sync --frozen --no-dev --extra forecast` and `CMD` is `python scripts/real_forecast.py`. That script loads WB2 HRES-T0, runs `run_rollout` on `aurora-finetuned`, and writes one `2t` PNG per step. Init is naive `datetime(2022, 6, 15, 12, 0)` — an aware UTC datetime raises a pandas index `TypeError`. CUDA fail-fast: the script returns 1 if `torch.cuda.is_available()` is false.

Weights are **not** baked in. Mount the host Hugging Face cache. The run needs network for GCS (`gs://weatherbench2`) and, on a cache miss, the fine-tuned checkpoint.

**Bind-mount `outputs/`.** `make docker-run-gpu` uses `--rm` and does **not** mount `/app/outputs`, so maps vanish with the container. Stage 1 used:

```bash
mkdir -p outputs ~/session-artifacts
docker run --gpus all \
  -e HF_HOME=/cache/huggingface \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$(pwd)/outputs:/app/outputs" \
  --rm \
  --platform "$PLATFORM" \
  aurora-inference:gpu-linux-amd64 \
  2>&1 | tee ~/session-artifacts/real_forecast.log
```

`--gpus all` is correct on a 1-GPU box. PNGs: `outputs/real_forecast_2t_stepNN.png` (gitignored).

Without Docker, on the instance:

```bash
uv sync --extra forecast --frozen
python scripts/real_forecast.py --steps 4
```

**HF Hub warning** (“set a HF_TOKEN”): optional for this public checkpoint. Never bake tokens into the image.

Capture host proof as well:

```bash
uname -m > ~/session-artifacts/uname.txt
nvidia-smi > ~/session-artifacts/nvidia-smi.txt
date -u > ~/session-artifacts/finished_utc.txt
```

---

## 7. Export artifacts (before teardown)

Run `scp` from your **laptop**, not from the instance.

```bash
# On the Mac (new local terminal) — instance still up:
mkdir -p ~/Downloads/lambda-session-$(date +%Y%m%d)
scp -r ubuntu@<INSTANCE_IP>:~/session-artifacts \
  ~/Downloads/lambda-session-$(date +%Y%m%d)/
scp -r ubuntu@<INSTANCE_IP>:~/ms_aurora_portfolio/outputs \
  ~/Downloads/lambda-session-$(date +%Y%m%d)/
```

Adjust the repo path if you cloned elsewhere. **Why not from the instance?** `scp ubuntu@<same-IP>:…` tries to SSH **into** the box again. The **private** key lives on your laptop.

Confirm files arrived locally before teardown.

---

## 8. TEARDOWN (mandatory)

1. `scp` done (section 7) — verify PNGs and logs on the laptop.
2. Lambda console → **Terminate** the instance — **not** Stop.
   Stopped instances can still bill; closing the laptop/Cursor tab does nothing.
3. Refresh until the instance is **gone**.
4. Confirm **instance** usage is no longer accruing.
5. If you attached a persistent filesystem: it **keeps billing** after terminate. Delete it at phase closeout (§0), or you pay ~$0.20/GiB-month on idle data.
6. Tick the DoD teardown boxes.

If the timebox ends mid-debug: **terminate anyway**. Resume later on a new box with a new written DoD.

---

## 9. Billing / spend (recap)

- Re-verify GPU **$/hr** and NFS **$/GiB-month** at every launch (both move).
- Use a hard timebox when Lambda has no in-console billing alarm.
- Stage 1 GPU session is done; do not leave a box up “for Stage 2 later.”
- NFS is a second bill: ~$0.20/GiB-month on **used** GiB, hourly increments, **including with no VM**. Dev splice ~$3/month; Aurora-only 2022 ~$80/month; full unused-var dump ~$110/month. Delete the volume when the phase ends.

---

## 10. Pitfalls checklist

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` not found; `lspci` shows the GPU | Plain Ubuntu image, no driver stack | Terminate; relaunch **GPU Base 22.04**. Do not apt-install `nvidia-utils-*`. |
| `permission denied … docker.sock` | `usermod` done but shell groups stale | `newgrp docker` or fresh SSH; or temporary `sudo docker` |
| `exec format error` at `uv sync` | Built default `linux/arm64` on x86 host | `make docker-build-gpu PLATFORM=linux/amd64` |
| `real_forecast.py` exits 1 immediately | No CUDA in the container | Confirm `--gpus all` and GPU Base image; `torch.cuda.is_available()` |
| PNGs gone after `docker run` | `--rm` without `/app/outputs` mount | Bind-mount `$(pwd)/outputs:/app/outputs` (README Quickstart) |
| `TypeError` comparing datetime to pandas index | Aware UTC `datetime` vs naive store index | Use naive `datetime(2022, 6, 15, 12, 0)` |
| HF “set a HF_TOKEN” warning | Unauthenticated Hub access | Optional; ignore if download succeeds |
| `scp` → `Permission denied (publickey)` | Ran `scp` *on* the instance toward itself | Run `scp` from the **Mac** |
| Nested `ms_aurora_portfolio/` on laptop | Accidental `git clone` inside local repo | `rm -rf` the nested copy; clone only on the instance |
| Cursor “port 22” toast | SSH session detected | Ignore — not the app server |
| NFS still billing after GPU terminate | Filesystem is independent of the instance | Delete the persistent filesystem in the console; confirm GiB-month usage stops |

---

## Quick reference — Stage 1 x86 GPU forecast

```text
DoD + timebox → launch A100 + GPU Base 22.04 (no persistent FS)
→ ssh ubuntu@IP → uname -m / nvidia-smi
→ usermod -aG docker + newgrp docker (or re-login)
→ docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi
→ clone → PLATFORM=linux/amd64 make docker-build-gpu
→ docker run --gpus all -v HF cache -v outputs --rm --platform linux/amd64 aurora-inference:gpu-linux-amd64
→ eyeball outputs/real_forecast_2t_stepNN.png
→ scp from Mac → TERMINATE → console gone
```

---

## Appendix — Stage 0 CPU-image rehearsal (historical)

Validated **2026-07-21** on 1× A10, GPU Base, `PLATFORM=linux/amd64`. Goal was lifecycle only: `make docker-build` / `make docker-run` → `scripts/synthetic_forward.py` (no skill, no PNGs). Do **not** use this as the Stage 1 forecast path.

```bash
export PLATFORM=linux/amd64
make docker-build PLATFORM="$PLATFORM"
make docker-run PLATFORM="$PLATFORM"
```
