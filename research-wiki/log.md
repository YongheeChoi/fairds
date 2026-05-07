# Research Wiki Log

_Append-only timeline._
- `2026-05-06T13:09:32Z` Wiki initialized
- `2026-05-06T13:11:23Z` ingest_paper: ingested paper:wang2024_data_shapley_one (arxiv:)
- `2026-05-06T13:11:28Z` ingest_paper: ingested paper:yan2022_forml_fairness_optimized (arxiv:)
- `2026-05-06T13:11:32Z` ingest_paper: ingested paper:arnaizrodriguez2023_fairshap_shapley_value (arxiv:)
- `2026-05-06T13:11:38Z` ingest_paper: ingested paper:ren2018_learning_reweight_examples (arxiv:)
- `2026-05-06T13:11:44Z` ingest_paper: ingested paper:lahoti2020_fairness_without_demographics (arxiv:)
- `2026-05-06T13:11:49Z` ingest_paper: ingested paper:cai2024_chg_shapley_singletrainingrun (arxiv:)
- `2026-05-06T13:15:47Z` ingest-plan: imported 6 papers, 6 claims, 7 experiments, 5 gaps, 1 idea from plan.md
- `2026-05-06T15:28:48Z` E1 completed: C3 partial (invalidated on toy LR setup; revisit on MLP+spurious / Adult-COMPAS), C1 invalidated (naive per-sample gradient overhead 11-30×; vectorize before E2/E4).
- `2026-05-06T15:43:04Z` E1b (spurious-feature MLP): C3 SUPPORTED (Mann-Whitney p=1.89e-08, Δφ monotone decreasing). E1 negative was LR setup limit — gap analysis for E2/E1b spurious confirms 2nd-order cross-term's auto-balancing in true representation-bias regime.
- `2026-05-06T16:10:23Z` E2 (Adult/COMPAS, head-to-head w/ Ren2018): C2/C4 INVALIDATED on real data. Ren2018 wins on fairness (worse accuracy); Fairds preserves accuracy but cannot match fairness improvement. Plan failure-mode triggered: paper requires reframe (workshop / efficiency-first / 1st-order-only).
- `2026-05-06T19:20:24Z` FINAL: 6 rounds of auto-review-loop with Codex GPT-5.2 → SCORE 6/10 reached. C3 SUPPORTED on E1b+E3, C2 partial (marginal head-to-head vs Ren2018 on E3), C2/C4 invalidated on Adult/COMPAS. Headline: Colored MNIST held-out test_worst fairds-2 +13.0pp vs vanilla, +2.7pp vs Ren2018 (p=0.052), 2nd-order isolation +4.7pp vs fairds-1 (p=0.033). Codex verdict: 'stop iterating'.
- `2026-05-07T00:11:20Z` Canonical re-review (gpt-5.5 xhigh): 6/10 'stop'. 5.2 verdict confirmed. Weak 6 (workshop↔main-track boundary). Damaging weakness: Colored MNIST only — Waterbirds 등 추가 시 진정 main-track.
- `2026-05-07T01:09:38Z` Round 8 (gpt-5.5 xhigh): SCORE 6/10 stable, 'stop'. 8-round full loop: R1=4, R2=5, R3=5, R4=6, R5=5, R6=6 (gpt-5.5 confirmed), R7=5 (Waterbirds catastrophic), R8=6 (regime-dependent pivot). Final paper framing: Colored MNIST positive + Waterbirds negative + honest scope statement.
- `2026-05-07T06:35:47Z` Stage 6 paper-writing: 3 review rounds (gpt-5.5 xhigh) — R1 4/10, R2 5/10, R3 5/10 stable weak accept. paper/main.pdf 4-page main body. E4 wall-clock: vanilla 1.08s/ep, fairds-1 4.30x, fairds-2 7.68x, ren2018 4.78x.
- `2026-05-07T06:49:24Z` Round 4 paper review with extended baselines (JTT, GroupDRO, IRM): 6/10 stable accept. GroupDRO best (0.900), JTT 2nd (0.878), Fairds-2 3rd (0.824) but isolation +4.7pp over Fairds-1 preserved. Honest reframing as mechanism paper.
- `2026-05-07T07:01:10Z` 🎯 R5 paper review: 7/10 ACHIEVED. Spurious-strength sweep (p=0.99 fairds-2 +55pp vs vanilla collapse) was the decisive evidence. Final state: research 6/10 + paper 7/10.
- `2026-05-07T07:38:23Z` R6 final stability check (gpt-5.5 xhigh): 7/10 stable, 'stop_at_7'. 7 is the ceiling — 8 requires second non-toy positive regime or deeper theory (out of scope). 17-round 진행 종료.
