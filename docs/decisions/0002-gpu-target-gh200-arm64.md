# ADR002: GPU target GH200 arm64

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07-21 |
| **Related** | Makefile `PLATFORM`; `docs/gpu-playbook.md`; Dockerfile dual-arch CPU path |

Sections follow the [Backstage ADR template](https://github.com/backstage/backstage/blob/master/docs/architecture-decisions/adr000-template.md) (Context / Decision / Consequences; Michael Nygard).

## Context

Local development is on an Apple Silicon Mac (`darwin/arm64`). Default Docker
CPU builds therefore target `linux/arm64` natively. The preferred cloud GPU for
later stages is a Lambda Labs GH200 (H100, 96 GB, host `aarch64`): same ISA
family as local images, previously validated as a host class (`nvidia-smi`,
NVIDIA Container Toolkit).

As of 2026-07, bookable GH200 / `aarch64` GPU stock is not consistently unavailable.
Common Lambda SKUs are `x86_64` (A10 / A100 / H100). Those are a different
host class: the same Dockerfile works only if `PLATFORM=linux/amd64` is set.
Silently equating GH200 and x86 hosts would cause `exec format error` failures
and confuse Stage 1 CUDA wheel work.

Stage 0 develops on CPU. A trial run on a cheap x86 GPU host (CPU image only, then terminate) is documented in
`docs/gpu-playbook.md`; the Dockerfile `gpu` stage remains unverified until
Stage 1.

## Decision

Lambda GH200 / `aarch64` is treated as the preferred GPU target when
stock returns, with Makefile defaulting to `PLATFORM=linux/arm64`.

`x86_64` Lambda SKUs (running on NVIDIA A10 / A100 / H100 etc.) as the working
fallback while GH200 is unavailable, using
`make docker-build PLATFORM=linux/amd64` (and matching `docker-run`).

We will re-check hourly rate, SKU availability, and `uname -m` at booking time.
We will not assume a CUDA base tag publishes both architectures without
`docker manifest inspect`.

We will keep one Dockerfile for both ISAs; `PLATFORM` selects the host class.
GPU image verification and CUDA wheel resolution remain Stage 1 work on the
booked arch.

## Consequences

- Image tags distinguish arches (`cpu-linux-arm64` vs `cpu-linux-amd64`); wrong
  `PLATFORM` on the host fails fast rather than looking like an app bug.
- The playbook and README must state preferred vs working hosts; ADR text is
  the record when catalog stock shifts.
- Stage 1 must resolve CUDA wheels and base-image manifests for the booked
  machine, not for an assumed GH200.
- When GH200 stock returns, the preferred path needs no redesign — only booking
  and `PLATFORM=linux/arm64`.
