---
type: experiment
node_id: exp:E3c
name: "Colored MNIST extended baselines (JTT, GroupDRO, IRM)"
status: completed
dataset: "Colored MNIST same setup as E3 (5000 train, ratio 0.9, anchor 200, val_eval 500, OOD test 5000), 10 seeds, 20 epochs, lr=0.05."
baseline: "Vanilla, Fairds-1, Fairds-2 (alpha=0.5, RMS-normalized), Ren2018 (corrected), JTT (lambda_up=5, 2-stage), GroupDRO (eta_q=0.01, n_groups=2), IRM (lambda=100 with anneal)."
added: 2026-05-07T06:49:23Z
---

# exp:E3c — Colored MNIST extended baselines (JTT, GroupDRO, IRM)

## Hypothesis

Comprehensive baseline coverage on the main positive benchmark (Colored MNIST). Adds JTT (no group label, 2-stage), GroupDRO (group label, minimax), IRM (group label, penalty) to the original 4-method comparison.

## Setup

- Dataset: Colored MNIST same setup as E3 (5000 train, ratio 0.9, anchor 200, val_eval 500, OOD test 5000), 10 seeds, 20 epochs, lr=0.05.
- Baseline: Vanilla, Fairds-1, Fairds-2 (alpha=0.5, RMS-normalized), Ren2018 (corrected), JTT (lambda_up=5, 2-stage), GroupDRO (eta_q=0.01, n_groups=2), IRM (lambda=100 with anneal).

## Success Criterion

GroupDRO best (test_worst 0.900±0.015) > JTT (0.878±0.058) > Fairds-2 (0.824±0.043) > Ren2018 (0.796±0.054) > Fairds-1 (0.777±0.071) > Vanilla (0.694±0.064) >> IRM (0.386±0.160 failed). Fairds-2 vs JTT: -5.4pp p=0.032 (JTT stronger in same supervision regime). Fairds-2 vs Fairds-1 isolation: +4.7pp p=0.033 (preserved). Paper paper now positions Fairds-2 as closed-form mechanism contribution rather than SoTA. Round 4 paper review: 6/10 stable accept (gpt-5.5 xhigh).

## Status

`completed` — updated by `/run-experiment` and `/result-to-claim`.

## Results

[Populated after run completes.]

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
