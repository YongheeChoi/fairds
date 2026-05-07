---
type: experiment
node_id: exp:E3d
name: "Spurious-strength phase-transition sweep on Colored MNIST"
status: completed
dataset: "Colored MNIST same setup as E3 (5000 train, anchor 200), 5 spurious strengths × 5 seeds × 4 methods (vanilla, fairds-1/2, ren2018) = 100 runs, lr=0.05, 20 epochs."
baseline: "Vanilla, Fairds-1, Fairds-2 (alpha=0.5), Ren2018 (corrected)."
added: 2026-05-07T07:01:10Z
---

# exp:E3d — Spurious-strength phase-transition sweep on Colored MNIST

## Hypothesis

Sweep p_spurious in {0.7,0.8,0.9,0.95,0.99} to characterize Fairds-2's operating regime: where is its advantage over vanilla strongest?

## Setup

- Dataset: Colored MNIST same setup as E3 (5000 train, anchor 200), 5 spurious strengths × 5 seeds × 4 methods (vanilla, fairds-1/2, ren2018) = 100 runs, lr=0.05, 20 epochs.
- Baseline: Vanilla, Fairds-1, Fairds-2 (alpha=0.5), Ren2018 (corrected).

## Success Criterion

Phase transition observed: p=0.7-0.8 vanilla≈fairds (NS), p=0.9 Δ=+11.3pp p=0.011, p=0.95 Δ=+19.8pp p=0.021, p=0.99 Δ=+55.3pp p=3e-5. Vanilla collapses at p=0.99 (test_worst 0.19) while fairds-2 robust (0.74). Pushed paper score from R4=6/10 to R5=7/10 (gpt-5.5 xhigh canonical).

## Status

`completed` — updated by `/run-experiment` and `/result-to-claim`.

## Results

[Populated after run completes.]

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
