---
type: experiment
node_id: exp:E3b
name: "Waterbirds (real-image spurious) — NEGATIVE on pretrained fine-tuning"
status: completed
dataset: "Waterbirds (Sagawa et al. 2020) — bird type × background spurious correlation, 4 groups, ImageNet-pretrained ResNet-18."
baseline: "Vanilla, Fairds-1, Fairds-2, Ren2018 (corrected) tried across 3 hp regimes: strong reweighting / warmup_epochs=3 / no_ema per-step."
added: 2026-05-07T01:09:38Z
---

# exp:E3b — Waterbirds (real-image spurious) — NEGATIVE on pretrained fine-tuning

## Hypothesis

Test C2/C3 on real-image spurious benchmark (Sagawa 2020) per gpt-5.5 xhigh Round 6 recommendation. ResNet-18 pretrained, BN frozen for vmap. Anchor 200 (50/group), val_eval 999, test 5794.

## Setup

- Dataset: Waterbirds (Sagawa et al. 2020) — bird type × background spurious correlation, 4 groups, ImageNet-pretrained ResNet-18.
- Baseline: Vanilla, Fairds-1, Fairds-2, Ren2018 (corrected) tried across 3 hp regimes: strong reweighting / warmup_epochs=3 / no_ema per-step.

## Success Criterion

C3/C2 NOT SUPPORTED on Waterbirds. fairds-2 best variant (no_ema, lr=1e-4, warmup=1, 15ep): test_worst=0.189 vs vanilla 0.115 (+7pp p=0.17 NS) vs ren2018 0.499 (Δ-31pp p=0.997 ren2018 dominates). All three algorithmic variants failed to approach ren2018. Honest negative reported per gpt-5.5 xhigh Round 7 pivot recommendation: regime-dependent mechanism paper. Evidence: results/e3b{,_warmup3,_no_ema}/<timestamp>/SUMMARY.md.

## Status

`completed` — updated by `/run-experiment` and `/result-to-claim`.

## Results

[Populated after run completes.]

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
