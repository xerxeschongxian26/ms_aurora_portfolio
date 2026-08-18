# ADR004: HRES_T0 Source of Record

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-05 |

Sections follow the [Backstage ADR template](https://github.com/backstage/backstage/blob/master/docs/architecture-decisions/adr000-template.md) (Context / Decision / Consequences; Michael Nygard).

## Context

To faciliate the goal of inference optimisation engineering, a benchmark set of results has to be obtained prior to performing any form of optimisation. This benchmark should approximate and even match previously reported performance figures from the original Aurora paper. An important criteria for selecting the performance metrics to be replicated was the absolute nature of the results i.e., the metrics cannot be reported relative to another model/set of results. e.g. relative RMSE. To that end, a set of performance figures to be matched was eventually identified from the Supplementary Information submitted with the original paper.

## Decision

The performance metrics identified are Figures H7, H8 and H9, shown in `docs/benchmark-target.md`. These metrics are absolute RMSE values for various predicted variables at different levels of the atmosphere. The underlying model was a fine-tuned model, specifically the Aurora 0.25 Fine-Tuned, which began as an AuroraPreTrained that was fine-tuned on the HRES_T0 dataset. Whilst the other models available had metrics reported, these were relative RMSE's, making their reproduction more costly in time and computational steps.

## Consequences

The underlying data source to be used in the inference optimisation will be HRES_T0.
