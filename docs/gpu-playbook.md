# GPU playbook (Lambda Labs)

Stage 0 still does **not** develop on a rented GPU. This file is the runbook for Stage 1+
sessions — and for an optional **lifecycle rehearsal** (spin up → configure → clone →
build → run → export → **terminate**) so teardown stays muscle memory.

**Standing rule (§7):** every paid session starts with a written definition of done and
ends with a **full terminate** (not stop). Forgotten instances destroy the budget.

Validated once on **2026-07-21**: 1× A10, Lambda **GPU Base 22.0**, `PLATFORM=linux/amd64`
CPU image build/run, artifacts `scp`'d home, then terminate. Lessons from that session
are folded into the sections below.

---

## 0. Before you click Launch

### Session definition of done (fill this in *before* launch)

Copy, edit, keep visible until teardown:

```
Date / timebox end: _______________  (hard stop — terminate even if DoD incomplete)
SKU booked: _______________________  (e.g. 1x A10 24GB)
Image: ____________________________  (must be GPU-ready — see §2)
ISA expected: x86_64 | aarch64
PLATFORM: linux/amd64 | linux/arm64
Hourly rate verified at booking: $_____ / hr
Billing alarm / spend guard: ________ (Lambda may not offer in-console alerts — use a
  timer + card limit / mental hard stop)

This session DoD (Stage 0 rehearsal — tick as you go):
  [ ] SSH as ubuntu; uname -m and nvidia-smi match the SKU
  [ ] Docker group active (newgrp or fresh login); GPU visible in container
  [ ] Clone repo; make docker-build PLATFORM=… succeeds (cpu target)
  [ ] make docker-run PLATFORM=… completes toy forward; shapes logged
  [ ] Artifacts copied off the box via scp from the laptop
  [ ] Instance TERMINATED; console shows gone; billing stopped

Out of scope this session:
  - Building/running the Dockerfile `gpu` stage (still scaffold / exit 1)
  - ERA5 / skill metrics / CUDA wheel debugging beyond toolkit smoke
  - Leaving the instance stopped overnight
  - apt-installing random nvidia-utils-* packages from Ubuntu's "can be installed with"
```

### Billing / spend control

- Re-check the **current hourly rate** and SKU availability at booking (both move).
- Lambda may **not** expose a first-class billing alarm — use a wall-clock timebox
  (e.g. 30–45 min for a Stage 0 rehearsal) and terminate when it ends.
- Prefer the cheapest bookable single-GPU x86 SKU for rehearsal (often **1× A10**).

### Persistent filesystem

For a short rehearsal: **do not** attach a Lambda persistent filesystem. Use the
instance local SSD only; `scp` artifacts home before terminate. Add a filesystem later
if Stage 1+ needs data that must survive across instances.

---

## 1. Target

| Preference | SKU | ISA | `PLATFORM` | When |
|---|---|---|---|---|
| **Preferred** | Lambda **GH200** (H100, 96 GB) | `aarch64` | `linux/arm64` | When in stock |
| **Working (2026-07)** | A10 / A100 / H100 on Lambda | `x86_64` | `linux/amd64` | GH200 unavailable |

Do **not** treat an x86 box as “the same as GH200.” Same Docker *recipe*, different
host class. Confirm with `uname -m` **on the instance** (not on your Mac).

| Where you run `uname` | Typical result |
|---|---|
| Mac (Apple Silicon) | `uname` → `Darwin`; `uname -m` → `arm64` |
| Lambda A10 / A100 / H100 | `uname` → `Linux`; `uname -m` → `x86_64` |
| Lambda GH200 | `uname` → `Linux`; `uname -m` → `aarch64` |

**Stage 0 rehearsal default:** cheapest bookable `x86_64` GPU. Goal is the lifecycle,
not throughput. Stage 0 uses the Dockerfile **`cpu`** image only.

Host CUDA ceiling on the previously validated GH200 class is **13.0** → keep container
CUDA in **12.4–12.8** when Stage 1 locks the GPU image.

---

## 2. Launch

1. Lambda Cloud console → create instance.
2. Pick region/SKU with stock; **write down $/hr**.
3. **Image (critical):** choose a **GPU-ready** image such as **GPU Base 22.0**
   (Ubuntu 22.04 family). Prefer **22.04** over 24.04 to stay aligned with
   `nvidia/cuda:*-ubuntu22.04` in the Dockerfile scaffold.
4. **Do not** pick a plain “Ubuntu 22.04 / 24.04” base for GPU work unless you
   already know it ships NVIDIA drivers. Lesson learned: plain Ubuntu on an A10
   still shows the GPU in `lspci`, but **`nvidia-smi` is missing** — do **not**
   `apt install nvidia-utils-*` from the distro hint list (wrong/old stacks).
   **Terminate** and relaunch with **GPU Base** instead.
5. Attach your SSH public key at create time.
6. Skip persistent filesystem for short sessions (§0).
7. Wait until running; copy the public IP.
8. Start a local timer for the timebox.

---

## 3. SSH

```bash
ssh ubuntu@<INSTANCE_IP>
```

User **must** be `ubuntu`. Lambda injects keys into `ubuntu`'s `authorized_keys`;
custom usernames fail.

If Cursor pops “application running on port 22”: that is the **SSH** port for this
session, not Aurora or Docker. Dismiss and continue in the terminal.

---

## 4. First-boot

```bash
uname -m          # expect: x86_64 (A10/A100/H100) or aarch64 (GH200)
nvidia-smi        # expect: the GPU you booked (A10, etc.)
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

**Lesson:** `usermod -aG docker` updates `/etc/group`, but Cursor/SSH sessions often
keep the old group set → `permission denied … docker.sock`. `newgrp docker` (or a
brand-new SSH login) fixes it. `sudo docker …` works as a temporary bypass; prefer
fixing the group so later `make docker-build` does not need `sudo`.

---

## 5. Toolkit regression (GPU visible in Docker)

On a new SKU this is a **regression check**. If it fails, stop before building
project images.

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi
```

Expect the booked GPU (e.g. A10) inside the container output.

| Flag / piece | Meaning |
|---|---|
| `--rm` | Delete the container when it exits |
| `--gpus all` | Pass host GPUs in (needs NVIDIA Container Toolkit) |
| image | CUDA 12.6.3 runtime on Ubuntu 22.04 |
| `nvidia-smi` | Command run *inside* the container |

Optional (Stage 1 lock-in later):

```bash
docker manifest inspect nvidia/cuda:12.6.3-runtime-ubuntu22.04
```

Confirm a manifest exists for the platform you will use. The Dockerfile `gpu` stage
still **`exit 1`** until Stage 1 — do not build `--target gpu` in Stage 0.

---

## 6. Clone, build, run (CPU image + HF cache)

### Match `PLATFORM` to the instance (mandatory)

| `uname -m` on instance | `PLATFORM` |
|---|---|
| `x86_64` | `linux/amd64` |
| `aarch64` | `linux/arm64` |

Makefile default is `linux/arm64` (GH200 / M1). On an A10, plain `make docker-build`
builds the **wrong** arch and fails at `uv sync` with:

```text
exec /bin/sh: exec format error
```

That is **not** “uv missing” — it is an **ISA mismatch** (e.g. arm64 `uv` binary on
x86). Always pass `PLATFORM` explicitly on cloud boxes.

```bash
git clone -b stage0-wp7 --single-branch \
  https://github.com/xerxeschongxian26/ms_aurora_portfolio.git
cd ms_aurora_portfolio
# After WP7 merges: clone default branch / main is fine.

export PLATFORM=linux/amd64   # on x86 Lambda; use linux/arm64 on GH200
make docker-build PLATFORM="$PLATFORM"
make docker-run PLATFORM="$PLATFORM"
```

What this does:

- Builds the Dockerfile **`cpu`** stage for that platform (tags
  `aurora-inference:cpu-linux-amd64` / `cpu-linux-arm64`).
- Runs `scripts/toy_forward.py` with `HF_HOME=/cache/huggingface` and
  `$HOME/.cache/huggingface` bind-mounted.
- May download the pinned HF checkpoint on first run — expected.

**HF Hub warning** (“unauthenticated requests… set a HF_TOKEN”): optional for this
public Aurora checkpoint. A token raises rate limits; skip unless you hit 429s or
very slow downloads. Never bake tokens into the image.

**Do not** `docker build --target gpu` in Stage 0.

### What `docker-run` produces

The toy forward **does not write project result files** — it logs shapes / timings /
peak RSS / the no-skill banner to **stdout**, then the container is removed (`--rm`).

What *does* persist on the instance:

- HF cache under `~/.cache/huggingface/` (via the mount)
- The Docker image you built

Capture session proof yourself:

```bash
mkdir -p ~/session-artifacts
make docker-run PLATFORM="$PLATFORM" 2>&1 | tee ~/session-artifacts/toy_forward.log
uname -m > ~/session-artifacts/uname.txt
nvidia-smi > ~/session-artifacts/nvidia-smi.txt
date -u > ~/session-artifacts/finished_utc.txt
```

| Command | Purpose |
|---|---|
| `mkdir -p ~/session-artifacts` | Folder to `scp` home |
| `… \| tee …/toy_forward.log` | Run again; show output and save log (`2>&1` includes warnings) |
| `uname -m > …` | Record instance ISA |
| `nvidia-smi > …` | Record GPU / driver snapshot |

---

## 7. Export artifacts (before teardown)

Run `scp` from your **laptop**, not from the instance.

```bash
# On the Mac (new local terminal) — instance still up:
mkdir -p ~/Downloads/lambda-session-$(date +%Y%m%d)
scp -r ubuntu@<INSTANCE_IP>:~/session-artifacts \
  ~/Downloads/lambda-session-$(date +%Y%m%d)/
```

**Why not from the instance?** `scp ubuntu@<same-IP>:…` tries to SSH **into** the
box again. The **private** key lives on your laptop; the instance only has your
**public** key → `Permission denied (publickey)`. Also `~/Downloads` on the instance
is not your Mac’s Downloads folder.

Confirm files arrived locally before teardown.

---

## 8. TEARDOWN (mandatory)

1. `scp` done (section 7) — verify files on the laptop.
2. Lambda console → **Terminate** the instance — **not** Stop.
   Stopped instances can still bill; closing the laptop/Cursor tab does nothing.
3. Refresh until the instance is **gone**.
4. Confirm usage is no longer accruing.
5. Tick the DoD teardown boxes.

If the timebox ends mid-debug: **terminate anyway**. Resume later on a new box with
a new written DoD.

---

## 9. Billing / spend (recap)

- Re-verify $/hr at every launch.
- Use a hard timebox when Lambda has no in-console billing alarm.
- One Stage 0 rehearsal is enough to validate this playbook; further GPU image work
  waits for Stage 1.

---

## 10. Pitfalls checklist (from 2026-07 rehearsal)

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` not found; `lspci` shows A10 | Plain Ubuntu image, no driver stack | Terminate; relaunch **GPU Base 22.0** (or equivalent). Do not apt-install `nvidia-utils-*`. |
| `permission denied … docker.sock` | `usermod` done but shell groups stale | `newgrp docker` or fresh SSH; or temporary `sudo docker` |
| `exec format error` at `uv sync` | Built default `linux/arm64` on x86 host | `make docker-build PLATFORM=linux/amd64` |
| HF “set a HF_TOKEN” warning | Unauthenticated Hub access | Optional; ignore if download succeeds |
| `scp` → `Permission denied (publickey)` | Ran `scp` *on* the instance toward itself | Run `scp` from the **Mac** |
| Nested `ms_aurora_portfolio/` on laptop | Accidental `git clone` inside local repo | `rm -rf` the nested copy; clone only on the instance |
| Cursor “port 22” toast | SSH session detected | Ignore — not the app server |

---

## Quick reference — Stage 0 x86 rehearsal

```text
DoD + timebox → launch A10 + GPU Base 22.0 (no persistent FS)
→ ssh ubuntu@IP → uname -m / nvidia-smi
→ usermod -aG docker + newgrp docker (or re-login)
→ docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi
→ clone -b stage0-wp7 → PLATFORM=linux/amd64 make docker-build && make docker-run
→ tee session-artifacts → scp from Mac → TERMINATE → console gone
```
