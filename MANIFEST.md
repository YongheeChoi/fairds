# Research Output Manifest

> Auto-maintained by KARIS skills. Tracks all generated artifacts across the research lifecycle.
> See [shared-references/output-manifest.md](../skills/shared-references/output-manifest.md) for the protocol.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-05-06 13:09 | research-wiki | research-wiki/ | init | Wiki 디렉토리 초기화 |
| 2026-05-06 13:15 | research-wiki | research-wiki/{papers,claims,experiments,ideas,gap_map.md,graph/edges.jsonl} | ingest-plan | plan.md → 6 papers / 6 claims / 7 experiments / 5 gaps / 1 idea / 20 edges |
| 2026-05-06 13:25 | research-pipeline | codes/fairds/{shapley,reweighter,trainer}.py | implementation | Fairds 코어 알고리즘 (1차/2차 Shapley + EMA + weighted SGD) |
| 2026-05-06 13:25 | research-pipeline | codes/experiments/e1_toy_2group/{data,run,analyze,ablate_alpha}.py | implementation | E1 실험 스크립트 |
| 2026-05-06 13:32 | run-experiment | codes/results/e1/20260506-132800/sweep_results.json | training | E1 full sweep (3 methods × 4 ratios × 5 seeds = 60 runs, 244s) |
| 2026-05-06 13:32 | analyze-results | codes/results/e1/20260506-132800/SUMMARY.md | analysis | E1 sweep 분석 — C3/C1 모두 NOT SUPPORTED |
| 2026-05-06 13:35 | run-experiment | codes/results/e1/ablate-alpha-20260506-133515/ablate_alpha.json | training | Alpha sweep (-2..+2, 3 seeds) — cross-term 부호 확인 |
| 2026-05-06 13:40 | research-wiki | research-wiki/claims/{C1,C3}.md, experiments/E1.md | review | 결과 반영 — C3 partial, C1 invalidated, E1 completed, 2 invalidates edges |
| 2026-05-06 15:34 | research-pipeline | codes/fairds/shapley.py | refactor | torch.func.vmap 적용 — per-sample gradient vectorize, 11-30× → 5-7× |
| 2026-05-06 15:34 | research-pipeline | codes/fairds/shapley.py::second_order_shapley_per_sample | refactor | Cross-term RMS-normalization 추가 — strong-spurious setup 에서 발산 방지 |
| 2026-05-06 15:36 | research-pipeline | codes/experiments/e1b_spurious_mlp/{data,run,analyze}.py | implementation | E1b: spurious-feature MLP 실험 (계획 외 추가 — C3 진짜 검증) |
| 2026-05-06 15:39 | run-experiment | results/e1b/20260506-153909/sweep_results.json | training | E1b full sweep (60 runs, 107s) |
| 2026-05-06 15:40 | analyze-results | results/e1b/20260506-153909/SUMMARY.md | analysis | E1b 분석 — **C3 SUPPORTED** (p=1.89e-08), Δworst fairds-2 vs fairds-1 일관 +0.7-1.3pp |
| 2026-05-06 15:40 | research-wiki | research-wiki/{experiments/E1b.md, claims/C3.md} | review | E1b 추가, C3 partial(LR-→) → partial(MLP+) (E2 검증 후 supported 승급 예정) |
| 2026-05-06 15:46 | research-pipeline | codes/datasets/{adult,compas}.py | implementation | UCI Adult + ProPublica COMPAS 데이터로더 (캐시, 50:50 balanced anchor 헬퍼) |
| 2026-05-06 15:48 | research-pipeline | codes/experiments/e2_adult_compas/{run,analyze,ablate}.py | implementation | E2 실험 스크립트 + 분석 + alpha/anchor ablation |
| 2026-05-06 15:55 | research-pipeline | codes/baselines/ren2018.py | implementation | Ren2018 1-step bi-level meta-learning (FORML 프로토타입) baseline |
| 2026-05-06 16:03 | run-experiment | results/e2/20260506-160342/sweep_results.json | training | E2 full sweep (vanilla/fairds-1/fairds-2/ren2018 × 2 datasets × 2 val-modes × 5 seeds = 80 runs, 292s) |
| 2026-05-06 16:04 | analyze-results | results/e2/20260506-160342/SUMMARY.md | analysis | E2 분석 — **C2 invalidated** (Ren2018 우위), **C4 invalidated** (recovery <90%), Fairds 는 accuracy-preserving 트레이드오프 |
| 2026-05-06 16:05 | research-wiki | research-wiki/{experiments/E2.md, claims/C2.md, claims/C4.md} | review | E2 추가, C2/C4 invalidated, edges 추가 |
| 2026-05-06 16:05 | research-pipeline | findings.md, CLAUDE.md, MANIFEST.md | review | 종합 요약 + 4 narrative 옵션 (A workshop / B efficiency / C 1차 only / D 재설계) |
| 2026-05-06 17:30 | auto-review-loop | NARRATIVE_REPORT.md (R1) | review | Round 1 codex review: SCORE 4/10, narrative reframe to mechanism paper |
| 2026-05-06 17:55 | research-pipeline | codes/{fairds/trainer.py, experiments/e2_adult_compas/{run.py,pareto_sweep.py,pareto_analyze.py}} | implementation | Round 1 fix: phi/w magnitude logging + Pareto sweep |
| 2026-05-06 17:58 | run-experiment | results/e2/pareto-20260506-175854/PARETO_SUMMARY.md | training | Pareto sweep on Adult (51 runs) — vanilla unique Pareto-optimal (later invalidated by Ren2018 fix) |
| 2026-05-06 18:00 | auto-review-loop | NARRATIVE_REPORT.md (R2 review) | review | Round 2: SCORE 5/10, Ren2018 1/B term bug 발견 |
| 2026-05-06 18:11 | research-pipeline | codes/baselines/ren2018.py | implementation | Round 2 fix: meta_loss = ((1/B)+eps)*per_sample.sum() — true 1-step bilevel |
| 2026-05-06 18:11 | run-experiment | results/e2/pareto-20260506-181137/PARETO_SUMMARY.md | training | Corrected Pareto sweep — vanilla AND ren2018/lr=0.05 둘 다 Pareto-optimal |
| 2026-05-06 18:15 | auto-review-loop | NARRATIVE_REPORT.md (R3 review) | review | Round 3: SCORE 5/10, Waterbirds/Colored MNIST 권장 |
| 2026-05-06 18:25 | research-pipeline | codes/{datasets/colored_mnist.py, experiments/e3_colored_mnist/{run.py,analyze.py}} | implementation | Round 3 fix: Colored MNIST 데이터 + run + analysis |
| 2026-05-06 18:30 | run-experiment | results/e3/20260506-183015/sweep_results.json | training | E3 first sweep (10 seeds, in-sample val_worst) |
| 2026-05-06 18:33 | auto-review-loop | NARRATIVE_REPORT.md (R4 review) | review | Round 4: SCORE 6/10 첫 도달! Ren2018 lr sweep 권장 |
| 2026-05-06 18:55 | run-experiment | results/e3/full-lr-sweep/* | training | All-method lr sweep × 10 seeds = 160 runs — fairds-2 best at lr=0.05 |
| 2026-05-06 19:00 | auto-review-loop | NARRATIVE_REPORT.md (R5 review) | review | Round 5: SCORE 5/10 재하락 — in-sample contamination 발각 |
| 2026-05-06 19:05 | research-pipeline | codes/datasets/colored_mnist.py | refactor | Round 5 fix: anchor / val_eval / test 3-way disjoint split + test groups |
| 2026-05-06 19:08 | run-experiment | results/e3/20260506-190605/SUMMARY.md | training | Held-out E3 sweep (lr=0.05, 10 seeds) — true OOD test_worst |
| 2026-05-06 19:10 | research-pipeline | NARRATIVE_REPORT.md | refactor | Round 5 cleanup: 모든 in-sample 수치 제거, single consistent headline |
| 2026-05-06 19:30 | auto-review-loop | research-wiki/, findings.md, CLAUDE.md, MANIFEST.md | review | **Round 6 final (gpt-5.2 high): SCORE 6/10 "stop iterating"** ✅ |
| 2026-05-07 09:00 | auto-review-loop | (verify) | review | **Canonical re-review with gpt-5.5 xhigh: 6/10 "6 confirmed, stop"** — 5.2 와 verdict 동의, weak 6 (workshop↔main-track 경계). 5.2 가 venue strength 에 살짝 lenient. |
| 2026-05-07 00:00 | research-pipeline | codes/datasets/waterbirds.py | implementation | Waterbirds (Sagawa 2020) 다운로드 + 데이터로더 (3-way split: 4795 train, 200 anchor, 999 val_eval, 5794 test, 4 groups) |
| 2026-05-07 00:15 | research-pipeline | codes/experiments/e3b_waterbirds/{run.py,analyze.py} | implementation | Waterbirds run + analysis (ResNet-18 pretrained, BN frozen for vmap) |
| 2026-05-07 00:25 | run-experiment | results/e3b/20260507-001647/sweep_results.json | training | Waterbirds 1차 (강 reweighting) — fairds catastrophic |
| 2026-05-07 00:38 | run-experiment | results/e3b/20260507-002752/SUMMARY.md | analysis | Waterbirds default-hp — fairds < vanilla « ren2018 |
| 2026-05-07 00:30 | research-pipeline | codes/fairds/trainer.py | refactor | warmup_epochs + no_ema 옵션 추가 (pretrained 대응 시도) |
| 2026-05-07 00:48 | run-experiment | results/e3b_warmup3/20260507-003831/SUMMARY.md | training | warmup_epochs=3 — fairds ≈ vanilla, ren2018 압도 |
| 2026-05-07 01:00 | auto-review-loop | (gpt-5.5 xhigh) | review | **R7 review: SCORE 5/10 (drop from R6's 6)** — Waterbirds negative cuts paper case. Pivot 권고. |
| 2026-05-07 01:07 | run-experiment | results/e3b_no_ema/20260507-005213/SUMMARY.md | training | no_ema (per-step) + lr=1e-4 — fairds-2 +7pp vs vanilla but p=0.17 NS, ren2018 -31pp p=0.997. |
| 2026-05-07 01:12 | auto-review-loop | (gpt-5.5 xhigh) | review | **R8 final review: SCORE 6/10 stable "stop"** — regime-dependent pivot 채택 후 점수 안정. 7점 fundamental 한계. |
| 2026-05-07 01:15 | research-pipeline | NARRATIVE_REPORT.md, findings.md, CLAUDE.md, MANIFEST.md, research-wiki/ | review | 8-round auto-review-loop 종료. 최종 paper framing: regime-dependent mechanism paper. |
| 2026-05-07 01:30 | paper-plan | PAPER_PLAN.md | paper-writing | NARRATIVE_REPORT.md → claims-evidence matrix + section structure (6 sections, 5 page workshop) |
| 2026-05-07 01:35 | run-experiment | results/e4/20260507-013330/sweep_results.json | training | E4 wall-clock overhead (CIFAR-10/ResNet-18): vanilla 1.08s/ep, fairds-1 4.30x, fairds-2 7.68x, ren2018 4.78x |
| 2026-05-07 01:36 | paper-figure | paper/figures/{fig_cmnist_results,fig_waterbirds,fig_e1b_phi,fig_pareto_adult}.pdf, paper/tables/{cmnist_main,cmnist_paired,waterbirds_grid}.tex | paper-writing | matplotlib + LaTeX figure/table 생성 |
| 2026-05-07 01:40 | paper-write | paper/{main.tex, sections/01..05.tex, sections/A_appendix.tex, references.bib} | paper-writing | LaTeX paper draft (4 page main body) |
| 2026-05-07 01:42 | paper-compile | paper/main.pdf (6 pages) | paper-writing | latexmk 성공 |
| 2026-05-07 01:43 | paper-review (gpt-5.5 xhigh) | (R1 review) | paper-review | **R1: 4/10 borderline workshop**, page budget over + credibility fixes 권고 |
| 2026-05-07 01:45 | paper-write | paper/sections/* (R1 fixes) | paper-writing | hero fig 제거, related compress, conclusion fold, wall-clock→appendix, CHG ref clean, 'no sensitive labels' wording, lr selection, mechanism soften |
| 2026-05-07 01:48 | paper-review (gpt-5.5 xhigh) | (R2 review) | paper-review | **R2: 5/10 weak accept** (+1) — mechanism softening incomplete |
| 2026-05-07 01:50 | paper-write | paper/sections/* (R2 fixes) | paper-writing | abstract verifies→supports, §4.2 To verify→probe, mechanism check→probe |
| 2026-05-07 01:52 | paper-review (gpt-5.5 xhigh) | (R3 final) | paper-review | **R3: 5/10 stable weak accept** — narrow positive evidence (Colored MNIST + MLP) limits ceiling. 7점 도달 fundamental 한계. |
| 2026-05-07 01:55 | research-wiki | research-wiki/{experiments/E4.md, claims/C1 edge}, findings.md, CLAUDE.md, MANIFEST.md | wrap-up | E4 + paper-writing 결과 wiki/findings/CLAUDE 반영. paper/main.pdf workshop submission-ready. |
| 2026-05-07 06:35 | research-pipeline | codes/baselines/{jtt,groupdro,irm}.py | implementation | JTT (Liu 2021), GroupDRO (Sagawa 2020), IRM (Arjovsky 2019) 베이스라인 구현 |
| 2026-05-07 06:43 | run-experiment | results/e3/20260507-063956/sweep_results.json | training | CMNIST extended sweep (7 methods × 10 seeds, 197s) |
| 2026-05-07 06:50 | paper-write | paper/{tables/cmnist_extended.tex, tables/cmnist_pairwise_f2.tex, figures/fig_cmnist_extended.pdf, sections/04_experiments.tex, sections/05_discussion.tex, references.bib} | paper-writing | Extended baseline 통합. Honest reframe: "closed-form mechanism, not SoTA" |
| 2026-05-07 06:55 | paper-review (gpt-5.5 xhigh) | (R4 review) | paper-review | **R4: 6/10 stable accept** (+1, R3=5) — extended baselines 효과 |
| 2026-05-07 06:50 | run-experiment | results/e3_strength/20260507-064949/sweep_results.json | training | Spurious-strength sweep (5 strengths × 5 seeds × 4 methods = 100 runs). Phase transition: p=0.99 에서 fairds-2 +55.3pp vs vanilla. |
| 2026-05-07 07:00 | paper-write | paper/{figures/fig_strength_phase.pdf, sections/A_appendix.tex (App C), sections/05_discussion.tex} | paper-writing | Phase transition appendix + 1-paragraph main body |
| 2026-05-07 07:05 | paper-review (gpt-5.5 xhigh) | (R5 final) | paper-review | **🎯 R5: 7/10 ACHIEVED** (+1, R4=6) — "phase-transition sweep pushes it over the 7 line" |
| 2026-05-07 07:10 | research-wiki | research-wiki/experiments/E3c.md, findings.md, CLAUDE.md, MANIFEST.md | wrap-up | 16 round 진행. Paper 7/10, Research 6/10. Workshop submission-ready |
| 2026-05-07 07:30 | paper-review (gpt-5.5 xhigh) | (R6 stability check) | paper-review | **R6 final: 7/10 stable, "stop_at_7"** — 7 is the ceiling. 8 requires new positive regime or theory. 17-round 진행 종료. |
| 2026-05-07 11:10 | paper-poster | poster/main.tex, poster/figures/ | poster-writing | A0 landscape ICLR-themed (deep green) poster — 4 columns, 11 cards |
| 2026-05-07 11:13 | paper-poster | poster/main.pdf (255 KB, 1 page A0) | poster-writing | latexmk 컴파일 성공 — title bar + 4 stat callouts (+13/+4.7/+55.3 pp/Mann-Whitney p) + Background/Method/Results/Phase/Waterbirds/Takeaways |
| 2026-05-07 11:14 | paper-poster | poster/poster_components.pptx (3 MB, 11 movable shapes), poster/poster.svg (830 KB) | poster-writing | PPTX (component-based, PowerPoint 편집 가능) + SVG (Illustrator 편집 가능) export |
| 2026-05-07 11:14 | paper-poster | poster/POSTER_SPEECH.md | poster-writing | 2-3분 발표 스크립트 + 6 anticipated Q&A |
