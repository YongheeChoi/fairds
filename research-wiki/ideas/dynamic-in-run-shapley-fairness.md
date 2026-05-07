---
type: idea
node_id: idea:dynamic-in-run-shapley-fairness
stage: proposed
based_on: ["paper:wang2024_data_shapley_one", "paper:yan2022_forml_fairness_optimized", "paper:arnaizrodriguez2023_fairshap_shapley_value", "paper:ren2018_learning_reweight_examples", "paper:lahoti2020_fairness_without_demographics", "paper:cai2024_chg_shapley_singletrainingrun"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3", "gap:G4", "gap:G5"]
added: 2026-05-06T13:13:24Z
---

# idea:dynamic-in-run-shapley-fairness

## Summary

Repurpose In-Run Data Shapley's closed-form Taylor approximations (1st: gradient dot-product, 2nd: gradient-Hessian-gradient cross-term) from a post-hoc data evaluation tool into a real-time bias-correcting dynamic controller. Anchored on a small demographically balanced unbiased validation set D_val, per-iteration Shapley values are computed via ghost-technique, accumulated via EMA, transformed into per-sample weights via softmax/clip, and applied as weighted SGD updates. This replaces FORML's bi-level meta-learning with closed-form Shapley derivation (zero implicit-differentiation cost), introduces meta-fairness as a guaranteed property, removes the sensitive-attribute requirement on D_val, and may automatically attenuate representation bias via the 2nd-order cross-term.

## Based On

paper:wang2024_data_shapley_one, paper:yan2022_forml_fairness_optimized, paper:arnaizrodriguez2023_fairshap_shapley_value, paper:ren2018_learning_reweight_examples, paper:lahoti2020_fairness_without_demographics, paper:cai2024_chg_shapley_singletrainingrun

## Target Gaps

gap:G1, gap:G2, gap:G3, gap:G4, gap:G5

## Stage

`proposed` — proposed → active → succeeded/failed/partial. Updated by `/result-to-claim`.

## Failure Notes

[Populated if stage transitions to failed/partial.]

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
