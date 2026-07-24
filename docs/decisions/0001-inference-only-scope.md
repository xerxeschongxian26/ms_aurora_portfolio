# ADR001: Inference-only scope

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07-21 |
| **Related** | Project thesis; README scope boundary; Stage 0 plan §3 |

Sections follow the [Backstage ADR template](https://github.com/backstage/backstage/blob/master/docs/architecture-decisions/adr000-template.md) (Context / Decision / Consequences; Michael Nygard).

## Context

This repository is a portfolio and engineering project built around Microsoft's
open Aurora weather foundation model. Upstream material — the original Aurora
paper, the Aurora 1.5 work, and Microsoft's official Aurora kit — emphasizes
model capability and fine-tuning.

A separate, thinner layer is self-hosted inference engineering: serving
weights reproducibly, validating inputs so Aurora cannot fail silently, and
later measuring how far cost and latency can be pushed before forecast fidelity
degrades past a defined tolerance. That layer is where this project is meant to
sit. Expanding into training would blur the thesis, raise compute cost, and
compete with Microsoft's documented fine-tuning story instead of complementing
it.

## Decision

I will keep this repository inference engineering only across all stages.

I will not add elements of training, fine-tuning, or distillation: no optimizers, no
loss functions, no `.backward()`, no training loops, no `DataLoader` feeding a
training step, and no LoRA tuning.

Fine-tuning will be treated as a future separate project, not a deferred
feature of this one.

## Consequences

- The README and roadmap remain a bounded niche (self-hosted
  inference and later optimization trade-offs) without competing as another
  fine-tuning demo.
- Stages 1–5 stay focused on the data layer, evaluation, serving, and inference
  optimization
- Contributors (including AI assistants) have an explicit stop rule when a
  change would pull in training concerns.
- Fine-tuning Aurora will not be performed and no training infrastructure will be built
