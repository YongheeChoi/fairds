# 프로젝트: fairds

## Pipeline Status

```yaml
stage: review  # 6-round auto-review-loop 종료, Codex SCORE 6/10 도달
idea: "Dynamic In-Run Data Shapley reweighting for fairness (idea:dynamic-in-run-shapley-fairness)"
contract: "plan.md"
current_branch: main
baseline: "E3 Colored MNIST held-out OOD test_worst: vanilla 0.694, ren2018 0.796, fairds-1 0.777, fairds-2 0.824 (best). 10 seeds × lr=0.05."
training_status: idle
language: ko
code_dir: codes
active_tasks: []
next: "/paper-writing 진입 가능. NARRATIVE_REPORT.md 가 input. 또는 추가 강화: Waterbirds 등 더 큰 spurious benchmark, larger-model overhead 측정 (C1)."
last_updated: "2026-05-06 19:30"
```

## 실험 종합 요약 (2026-05-06, 6-round auto-review 완료)

### Claims 상태 (final)
| claim | E1 LR | E1b spurious MLP | E2 Adult/COMPAS | E3 Colored MNIST (held-out OOD) | 종합 |
|---|---|---|---|---|---|
| **C1** (overhead ≤1.5×/3×) | 11-30× ❌ | 5-7× (vmap) | 5-7× | 5-7× | partial (large-model 측정 미완료) |
| **C2** (FORML 동등+) | n/a | n/a | dominated ❌ | **+2.7pp p=0.052** marginal-supported | **partial** |
| **C3** (2차 자동균형) | not (LR 한계) | **SUPPORTED** p=1.89e-08 | marginal | **SUPPORTED** isolation +4.7pp p=0.033 | **supported** |
| **C4** (sensitive-free) | n/a | n/a | recovery <90% ❌ | n/a | invalidated on tabular |
| C5/C6 | 미검증 | 미검증 | 미검증 | n/a | n/a |

### 6-round Codex auto-review-loop 진행 (gpt-5.2 high, R1-R6)
- R1: 4/10 → narrative reframe + Pareto sweep
- R2: 5/10 → Ren2018 baseline 1/B term fix
- R3: 5/10 → Colored MNIST 추가
- R4: 6/10 → lr sweep best-tuned 비교
- R5: 5/10 (재하락) → in-sample contamination 발각, held-out split 필요
- **R6: 6/10 reached, "stop iterating"** ✅

### Canonical KARIS final review (gpt-5.5 xhigh, 2026-05-07)
- **6/10 "6 confirmed, stop"** ✅
- 5.2 와 verdict 동의, 단 "weak 6" — workshop↔main-track 경계
- gpt-5.2 가 venue strength 에 살짝 lenient 했음
- 가장 큰 약점: Colored MNIST 만으로 narrow → Waterbirds 등 더 큰 spurious benchmark 가 main-track 진정 보장

### Round 7-8: Waterbirds 추가 + pivot (gpt-5.5 xhigh, 2026-05-07)
- **R7: 5/10** — Waterbirds 에서 fairds catastrophic 패배 (3 hp variant 모두 ren2018 못 따라잡음)
- **R8 final: 6/10 stable, "stop"** — regime-dependent pivot 채택 후 점수 안정
- 7점 fundamental 한계: pretrained ResNet fine-tuning 환경에서 fairds 알고리즘 약점 명확
- 최종 paper framing: from-scratch / small-model 에서 mechanism 입증 + pretrained 에서 honest negative
- 8 라운드 research review 진행 종료

### Stage 6 paper-writing (2026-05-07)
- `/paper-writing` 호출 → PAPER_PLAN.md → figures/tables → paper/main.pdf (4 main body + refs/appendix)
- 5-round paper review (gpt-5.5 xhigh): R1 4/10 → R2 5/10 → R3 5/10 → R4 6/10 (baselines) → **R5 7/10** (phase-transition sweep)
- 추가 baseline (R4): JTT, GroupDRO, IRM — Colored MNIST 7 methods × 10 seeds. JTT/GroupDRO 더 강함 (정직 보고)
- 추가 sweep (R5): spurious_strength {0.7, 0.8, 0.9, 0.95, 0.99} — fairds-2 vs vanilla 의 phase transition. p=0.99 에서 **+55.3pp (p=3e-5)** 결정적
- E4 (wall-clock overhead): vanilla 1.08s/ep, fairds-1 4.30×, fairds-2 7.68×, ren2018 4.78×
- **🎯 7점 도달**: Codex verdict "phase-transition sweep pushes it over the 7 line"
- Workshop submission-ready (ICLR_workshop level), borderline main-track candidate

### 최종 점수
- **Research narrative (R8 final, gpt-5.5 xhigh)**: 6/10 stable
- **Paper (R5 → R6 final stability check, gpt-5.5 xhigh)**: **7/10 stable, "stop_at_7"** ✅
- 17 round 진행 종료 (8 research + 6 paper + 3 implementation rounds)
- **R6 verdict**: "7 is the ceiling for this scope. A true 8 would need a second convincing non-toy positive regime or substantially deeper theoretical result."

### Headline result
**Colored MNIST 표준 spurious benchmark, held-out OOD test_worst:**
- vanilla 0.694 → fairds-2 0.824 (Δ +13.0pp, p=5e-5)
- vs corrected Ren2018: Δ +2.7pp (p=0.052)
- 2nd-order isolation vs fairds-1: Δ +4.7pp (p=0.033)
- 모든 method 중 lowest seed variance (test std 0.039)

### 자세히
- `NARRATIVE_REPORT.md` — paper-ready narrative
- `results/e3/20260506-190605/SUMMARY.md` — main result table
- `results/e1b/20260506-153909/SUMMARY.md` — mechanism evidence
- `results/e2/{20260506-160342, pareto-20260506-181137}/` — tabular failure analysis
- `findings.md`, `research-wiki/`

## E1 결과 요약 (2026-05-06)

- **C3 NOT SUPPORTED on LR:** 다수 그룹 평균 phi 가 항상 *큼* (Δφ +0.00026 ~ +0.00145, ratio 증가에 따라 *증가*). 1차/2차 격차 ≈ 3e-5. → LR Hessian 정보 부족 + toy 가 진짜 representation bias 가 아닌 셋업.
- **C1 (구현 후 재측정 필요):** Naive 11–30×. vmap 적용 후 5.3×/6.7× (hidden=64 MLP). 큰 모델 (ResNet) 에서 비율 줄어들 것.

## E1b 결과 요약 (2026-05-06) — C3 가설 입증 🎯

- **C3(a) SUPPORTED**: Mann-Whitney U(maj<min) p=**1.89e-08** at ratio=0.9 for fairds-2.
- **C3(b) SUPPORTED**: Δφ monotone decreasing (0.50→+0.002, 0.99→-0.025).
- **2차 isolation**: fairds-2 worst-acc +0.7-1.3pp 일관되게 fairds-1 우위.
- **알고리즘 보완**: cross-term RMS-normalized 필수 (그렇지 않으면 strong-spurious setup 에서 발산).
- 자세히: `results/e1b/20260506-153909/SUMMARY.md`, `findings.md`, `research-wiki/{claims/C3.md, experiments/E1b.md}`

## 프로젝트 제약

- (미정 — 추후 채워주세요)

## 비목표 (Non-Goals)

- (미정)

## 컴퓨팅 예산

- rtx p0 x2, no time limit
<!-- KARIS:BEGIN -->
## KARIS Skill Scope
KARIS skills installed in this project: 55 entries.
Manifest: `.karis/installed-skills.txt` (lists every skill KARIS installed and its upstream target).
For KARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/users/yonghee/projects/karis`).
Update with: `bash /home/users/yonghee/projects/karis/tools/install_karis.sh`  (re-runnable; reconciles new/removed skills).
<!-- KARIS:END -->
