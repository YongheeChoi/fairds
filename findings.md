# Findings

> **Cross-stage discovery log.** Records what you learn during experiments — both research insights about your method/claims and engineering lessons from debugging. Read on every session recovery, so keep entries concise.
>
> **Why this file exists:** Experiments produce discoveries that are critical for future decisions but don't belong in formal experiment reports. Without a central log, these get lost between sessions — and the next session repeats the same mistakes or misses important signals.

---

# Research Findings

> Method-level insights: what works, what doesn't, and why. These directly inform your claims, experiment design, and paper narrative.

## [2026-05-06] E1 — 2차 Shapley 의 자동 균형 가설 (C3) 1차 검증 실패
- **결과:** Toy 2-group LR (n=2000, ratio∈{0.5, 0.7, 0.9, 0.99}, 5 seeds) 에서 fairds-2 의 다수 그룹 평균 phi 가 소수 그룹보다 *항상 큼*. ratio=0.9 에서 Mann-Whitney U(maj < min) p=1, 즉 가설과 반대 방향. Imbalance 증가 시 Δφ 도 단조 증가 (0.50→+0.00026, 0.99→+0.00145).
- **1차 vs 2차 격차 ≈ 0:** Δφ₂ − Δφ₁ ≈ −3e-5. Cross-term 의 실효 영향이 거의 0.
- **alpha sweep (-2 ~ +2, 3 seed):** 부호 어떻게 잡아도 Δφ > 0. 음수 alpha 가 약간 균형 회복 (Δφ -2.0=+0.00100 vs +2.0=+0.00140) 하나, 1차 baseline (alpha=0, Δφ=+0.00122) 자체가 양수라서 cross-term 으로 충분히 상쇄 안 됨.
- **해석:** 단순 implementation bug 가 아니라 셋업 문제로 보임. (1) Logistic regression 의 Hessian 은 input feature × feature 만 — 깊은 모델의 representation Hessian 과 다름. (2) 우리 toy의 "bias" 는 그룹별 노이즈 분산 차이 — 진정한 representation bias (spurious correlation, e.g. background leak) 가 아님.
- **결정:** plan.md 의 failure mode 발동 직전 상태. 결정 전 추가 검증: (a) 2-layer MLP + spurious feature toy, (b) Adult/COMPAS (E2) 직행. 두 결과 모두 부정이면 plan 의 "1차 only paper 로 후퇴" 경로 채택.
- **Evidence:** `codes/results/e1/20260506-132800/SUMMARY.md`, `codes/results/e1/ablate-alpha-20260506-133515/ablate_alpha.json`

## [2026-05-06] E1 — Fairness metric 변화 미미
- **결과:** Toy 데이터에서 vanilla / fairds-1 / fairds-2 모두 acc 0.974-0.978, DP-diff 0.011-0.022, EO-diff 0.046-0.052. 사실상 동일.
- **원인:** Toy 가 너무 쉬워 (결정 경계가 그룹과 독립) Vanilla LR 도 acc 0.97+ 수렴. Reweighting 의 여지가 거의 없음.
- **함의:** E1 만으로는 C2 (FORML 대비 동등 이상 fairness) 검증 불가. Adult/COMPAS (E2) 에서 본격 검증 필요.

## [2026-05-07] 🎯 Round 4-5 paper review — SCORE 7/10 achieved! (target reached)
- **R4 (extended baselines)**: 6/10 stable accept. JTT, GroupDRO, IRM 추가 후 더 honest+credible.
- **R5 (spurious-strength sweep)**: **7/10 achieved**. Phase-transition 결과 결정적:
  - p=0.70-0.80: vanilla ≈ fairds (NS)
  - p=0.90: Δ=+11.3pp (p=0.011)
  - p=0.95: Δ=+19.8pp (p=0.021)
  - **p=0.99: Δ=+55.3pp (p=3e-5)** — vanilla collapse (0.19), fairds-2 robust (0.74)
- **Codex verdict**: "phase-transition sweep pushes it over the 7 line. The +55.3pp at p_majority=0.99 is genuinely compelling. Changes the story from 'Fairds-2 helps on one Colored MNIST setting' to 'Fairds-2 has a sharply defined operating regime where vanilla collapses and closed-form reweighting remains robust.'"
- **Stronger empirical shape** + honest baseline reporting + 2nd-order isolation 보존.
- **Evidence**: `paper/main.pdf` (7 pages), `paper/figures/fig_strength_phase.pdf`, `results/e3_strength/20260507-064949/`

## [2026-05-07] 🔬 E3c — Extended baselines (JTT, GroupDRO, IRM) on Colored MNIST
- 7 methods × 10 seeds × lr=0.05 × 20 epochs:
  - vanilla 0.694, fairds-1 0.777, **fairds-2 0.824**, ren2018 0.796, jtt 0.878, **groupdro 0.900**, irm 0.386 (failed)
- Fairds-2 vs JTT (same supervision regime): -5.4pp (p=0.032, JTT better)
- Fairds-2 vs GroupDRO (different supervision): -7.6pp (p=9e-5, GroupDRO better)
- Fairds-2 vs Fairds-1 isolation: +4.7pp (p=0.033) **유지**
- Paper reframed: "closed-form mechanism contribution, not SoTA" — JTT/GroupDRO 정직 보고
- **Evidence**: `results/e3/20260507-063956/SUMMARY.md`

## [2026-05-07] 📄 Stage 6 — `/paper-writing` 완료, ICLR Workshop draft (5/10 weak accept)
- **셋업**: NARRATIVE_REPORT.md → PAPER_PLAN.md → figures/tables (matplotlib) → LaTeX paper (4 page main body + 2 page refs/appendix) → 3 round gpt-5.5 xhigh review.
- **3-round review**: R1 4/10 borderline → R2 5/10 weak accept (+1, page budget+credibility fixes 적용) → R3 5/10 stable weak accept (mechanism softening 추가).
- **사용자 목표 7점 미달**: positive evidence 가 narrow (Colored MNIST + 합성 MLP 만). non-toy positive benchmark 추가 필요하나 시간 비용 큼.
- **paper/main.pdf**: 6 pages 총, 4 page main body. Workshop 제출 가능.
- **Evidence**: `paper/main.pdf`, `paper/sections/*.tex`, `paper/references.bib`

## [2026-05-07] 📊 E4 — Wall-clock overhead measurement (CIFAR-10 / ResNet-18)
- **셋업**: 10000-sample CIFAR-10 subset, ResNet-18 from scratch (BN frozen for vmap), batch 256, 3 epochs × 3 seeds.
- **결과 (sec/epoch, ratio vs vanilla)**:
  - vanilla 1.08±0.05 (1.00×)
  - fairds-1 4.63±0.01 (**4.30×**)
  - **fairds-2** 8.27±0.02 (**7.68×**)
  - ren2018 (corrected) 5.15±0.02 (**4.78×**)
- **C1 (target 1.5×/3.0×) NOT SUPPORTED**, 단 **fairds 가 bi-level baseline 과 비교 가능**:
  - fairds-1 < ren2018 (4.30× < 4.78×) — closed-form 1차가 bi-level 보다 빠름
  - fairds-2 > ren2018 (7.68× > 4.78×, 1.6× 더 느림)
- **함의**: closed-form Shapley 의 efficiency advantage 는 1차에선 명확, 2차에선 vmap+HVP 비용으로 bi-level 에 밀림. Paper Appendix A 에 정직 보고.
- **Evidence**: `results/e4/20260507-013330/sweep_results.json`

## [2026-05-07] 🚨 E3b — Waterbirds (Sagawa 2020): fairds 명확히 패배, regime-dependent pivot
- **셋업**: ResNet-18 pretrained (BN frozen for vmap), 4795 train, 200 anchor (50/group), 999 val_eval, 5794 test (4 groups). ImageNet weights, 96x96 resize.
- **3 알고리즘 variant 모두 실패**:
  - 강 reweighting (τ=0.1, ws=4): fairds-2 test_worst 0.135±0.063 (catastrophic, vanilla 0.182 보다도 떨어짐)
  - default + warmup_epochs=3: fairds-2 0.200±0.086 (≈ vanilla, p=0.41 NS)
  - **no_ema per-step + lr=1e-4 + warmup=1, 15ep**: fairds-2 0.189±0.047 (vs vanilla +7pp p=0.17 NS)
- **ren2018 (corrected) 압도적**: test_worst 0.494-0.499±0.06 — fairds-2 보다 +29-35pp 우위 (p<0.005)
- **진단**: Pretrained model 의 fine-tuning 환경에서 g_i 가 small/noisy → reweighting 이 학습 dynamics 망김. 3 가지 fix 시도 모두 ren2018 따라잡기 실패.
- **High variance**: seed=2 가 모든 fairds 변형에서 collapse (학습 dynamics 의 구조적 fragility).
- **Round 8 final pivot**: regime-dependent mechanism paper. Practical recommendation: pretrained fine-tuning 환경에선 bi-level meta-learning (Ren2018) 사용.
- **Evidence**: `results/e3b/20260507-002752/`, `results/e3b_warmup3/20260507-003831/`, `results/e3b_no_ema/20260507-005213/SUMMARY.md`

## [2026-05-06] 🎯 E3 — Colored MNIST: 핵심 가설 입증 (held-out OOD), Codex review SCORE 6/10
- **셋업:** 표준 spurious-correlation benchmark (Arjovsky 2019). n_train=5000, P(c=y|maj)=0.9, P(c=y|min)=0.5, P(c=y|test)=0.1. anchor (200, 학습 중 사용) / val_eval (500, held-out) / test (5000, OOD with test groups = color-aligned vs flipped). 10 seeds, lr=0.05 best-tuned per method. Small CNN.
- **headline (held-out OOD test_worst):**
  - vanilla 0.694, ren2018 (corrected) 0.796, fairds-1 0.777, **fairds-2 0.824**
- **fairds-2 vs vanilla**: Δtest_worst = **+13.0pp (p=5e-5)** ⭐⭐
- **fairds-2 vs corrected Ren2018**: Δtest_worst = +2.7pp (p=0.052 one-sided), Δtest_acc=+2.3pp (p=0.057). Marginal but consistent.
- **2nd-order isolation (fairds-2 vs fairds-1)**: Δtest_worst = +4.7pp (p=0.033). 통계적으로 유의한 cross-term 효과.
- **Variance**: fairds-2 가 모든 method 중 가장 stable (test std 0.039 vs ren2018 0.048, vanilla 0.066).
- **Codex GPT-5.2 6-round auto-review**: 4→5→5→6→5→**6 reached, "stop iterating"**.
- **Evidence**: `results/e3/20260506-190605/SUMMARY.md`, `NARRATIVE_REPORT.md`

## [2026-05-06] E2 — Adult/COMPAS, head-to-head with Ren2018: C2/C4 INVALIDATED
- **셋업:** UCI Adult (MLP, 36k samples, sex sensitive) + ProPublica COMPAS (LR, 4k samples, race sensitive). 5 seeds, 2 val modes (balanced 50:50 vs random subsample), epochs=15, lr=0.01, alpha=0.5. 베이스라인: Vanilla, Fairds-1, Fairds-2, **Ren2018 (FORML 프로토타입, 1-step bi-level)**.
- **C2 NOT SUPPORTED**: Adult 에서 Ren2018 dp_diff 0.175 / eo_diff 0.102 (vanilla 0.196/0.117 대비 -2.1pp/-1.5pp 개선). Fairds-2 dp 0.189 / eo 0.127 (-0.7pp/+1.0pp). **Fairds-2 가 Ren2018 보다 양 fairness metric 모두 1.4-2.5pp 뒤쳐짐**. COMPAS 도 비슷 (Fairds-2 vs Ren2018: dp+0.8pp, eo+0.3pp 손해).
- **다른 operating point**: Fairds-2 의 강점은 accuracy 보존 (Adult Δacc -0.25pp), Ren2018 은 fairness 우선 (Δacc -1.84pp). 둘은 같은 트레이드오프 곡선의 다른 지점.
- **C4 NOT SUPPORTED**: random vs balanced anchor 의 dp_diff recovery ratio: Adult 32% (Fairds-2), COMPAS -414% (의미없음). 90% 목표 명백히 미달. EO recovery 도 -60% 등 부정.
- **Mechanism 자체는 작동**: phi-by-group 으로 보면 Fairds 는 양 데이터셋 모두 majority (Adult=male, COMPAS=Black) 를 down-weight. 하지만 reweighting magnitude (Δw ≈ 0.01) 가 너무 작아 fairness metric 으로 안 변환.
- **Alpha sweep on Adult**: alpha ∈ {0.5, 1.0, 2.0, 4.0} × anchor ∈ {200, 500, 1000}. alpha≥2.0 시 학습 collapse (acc 0.256). alpha 1.0 + anchor 1000 정도가 최선이나 vanilla 와 사실상 동등.
- **함의:** plan.md C2 success criterion ("Fairds 가 FORML 대비 두 fairness 지표 모두에서 동등 이상") **명확히 실패**. 이건 plan 의 *failure mode* — "C2 unsupported 시 → 이론 차별만으로는 ML 학회 통과 어려움 → 실험 설계 또는 알고리즘 재검토 필요". 가능 옵션: (A) workshop reframe with E1b 강조, (B) efficiency-first reframe, (C) 1차 only retreat, (D) cross-term scaling 재설계.
- **Evidence:** `results/e2/20260506-160342/SUMMARY.md`

## [2026-05-06] E1b — C3 가설 입증! (spurious-feature 2-layer MLP setup)
- **셋업:** E1 의 LR 한계 진단 후 새로 설계 — 다수 그룹의 spurious feature z 가 label 과 강한 상관 (s=10.0), 소수 그룹은 z 무관, 진짜 informative feature 는 약함 (class_separation=0.5). MLP 가 z 에 의존 → minority acc 무너짐. 진짜 representation bias 시뮬레이션.
- **C3(a) SUPPORTED**: Mann-Whitney U(maj<min) p=**1.89e-08** at ratio=0.9 for fairds-2. 다수 그룹 phi 가 소수 그룹보다 *유의하게 작음*.
- **C3(b) SUPPORTED**: Δφ monotone *decreasing* (0.50→+0.002, 0.70→-0.003, 0.90→-0.011, 0.99→-0.025) — imbalance 클수록 majority 더 강하게 penalty.
- **2차의 isolation effect**: fairds-2 worst-group acc 가 fairds-1 보다 일관 +0.7-1.3pp. 1차만으론 weight 가 너무 극단적 → 학습 안정성 깨짐 → minority acc 못 올림. 2차 cross-term 이 *부드럽게* reweighting (Δw 작음) 하여 stability 유지.
- **vs vanilla**: fairds-2 worst-acc +0.4-1.2pp (ratio 0.9 에서 p=0.052, 5 seed). Effect size 작지만 일관됨.
- **결정적 함의:** E1 negative 는 LR 의 Hessian 정보 부족 + toy noise-difference setup 한계였음. **plan.md 의 핵심 가설 (G5/C3) 가 진짜 representation-bias regime 에서 입증.** E2 Adult/COMPAS 직행 권고.
- **알고리즘 보완**: cross-term 을 RMS-normalized → first-order 와 동일 magnitude 로 맞춤. 이게 안 되면 spurious 강한 setup 에서 fairds-2 발산 (smoke 1차 시 acc 0.5 random).
- **C1 재측정**: vmap 적용 후 vanilla 0.41s, fairds-1 2.18s (5.3×), fairds-2 2.76s (6.7×) at hidden=64. 큰 모델일수록 vanilla 부담 증가 → 비율 줄어들 것 (CIFAR ResNet-18 에서 재측정 필요).
- **Evidence:** `results/e1b/20260506-153909/SUMMARY.md`, `codes/fairds/shapley.py::second_order_shapley_per_sample`

---

# Engineering Findings

> Infrastructure, environment, and debugging lessons. Prevents re-debugging the same issues in future sessions.

## [2026-05-06] Per-sample gradient overhead — vectorization 미적용 시 11–30×
- **문제:** Naive Python loop 으로 sample-by-sample gradient 계산 시 vanilla LR 대비 wall-clock 11×~30× (n=2000, batch=64). C1 의 1.5×/3.0× 목표 위배.
- **원인:** `torch.autograd.grad` 를 batch 내 i 별로 별도 호출 — 그래프 재구성 비용이 가장 큼.
- **해결책 (E2/E4 전 적용 필요):** `torch.func.vmap(grad)` 또는 `torch.func.functional_call` 로 per-sample gradient vectorize. Wang et al. (2024) 의 ghost technique 도 마찬가지로 vectorized — 그대로 사용 가능.
- **적용 위치:** `codes/fairds/shapley.py::_per_sample_gradients`
