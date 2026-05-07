---
type: experiment
node_id: exp:E1b
name: "Spurious-feature 2-group MLP"
status: completed
dataset: "Synthetic 2-group: majority has spurious feature z perfectly correlated with y (s=10.0), minority has z independent. d_core=8 with 2 informative dims, class_separation=0.5. n=1500, n_val=300 balanced. ratios {0.5,0.7,0.9,0.99}, 5 seeds."
baseline: "Vanilla MLP (hidden=64, 2-layer ReLU), Fairds-1st, Fairds-2nd with RMS-normalized cross-term"
added: 2026-05-06T15:42:43Z
---

# exp:E1b — Spurious-feature 2-group MLP

## Hypothesis

Re-test C3 in a setup with TRUE representation bias (spurious feature held only in majority group, weak true signal). Tests whether E1's negative was due to LR's information-poor Hessian or to a fundamental gap in the 2nd-order auto-balancing claim.

## Setup

- Dataset: Synthetic 2-group: majority has spurious feature z perfectly correlated with y (s=10.0), minority has z independent. d_core=8 with 2 informative dims, class_separation=0.5. n=1500, n_val=300 balanced. ratios {0.5,0.7,0.9,0.99}, 5 seeds.
- Baseline: Vanilla MLP (hidden=64, 2-layer ReLU), Fairds-1st, Fairds-2nd with RMS-normalized cross-term

## Success Criterion

C3 SUPPORTED — Mann-Whitney U(maj<min) p=1.89e-08 at ratio=0.9; Δφ monotone decreasing across ratios for fairds-2 (0.50→+0.002, 0.99→-0.025); fairds-2 worst-group acc consistently above fairds-1 by +0.7-1.3pp. Vanilla overhead ratio 5-7× (vs E1's 11-30×). Evidence: results/e1b/20260506-153909/SUMMARY.md.

## Status

`completed` — updated by `/run-experiment` and `/result-to-claim`.

## Results

[Populated after run completes.]

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
