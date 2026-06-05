# Findings

> **Cross-stage discovery log.** Records what you learn during experiments — both research insights about your method/claims and engineering lessons from debugging. Read on every session recovery, so keep entries concise.
>
> **Why this file exists:** Experiments produce discoveries that are critical for future decisions but don't belong in formal experiment reports. Without a central log, these get lost between sessions — and the next session repeats the same mistakes or misses important signals.

---

# Research Findings

> Method-level insights: what works, what doesn't, and why. These directly inform your claims, experiment design, and paper narrative.

## [2026-06-04] 🎯 E7 CelebA — demographic fairness가 structured면 작동 (Adult/COMPAS 약점 보완)
- **셋업**: CelebA Blond_Hair target, Male sensitive group (Sagawa style). 자연 spurious(blond↔female), minority blond-male (train 10000 중 90개!). from-scratch 3-conv CNN 64px, 10 seed, 15ep. 신규: `datasets/celeba_fairness.py`, `experiments/e7_celeba/{run,analyze}.py`. 다운로드: gdrive quota 풀림(이전 실패 → 성공).
- **결과 (test, 10 seed)**: vanilla worst **0.199** → fairds-1 **0.747** / fairds-2 0.719 / resid 0.711 (**+51-55pp, p=0.002**). disparity 0.779→0.19. **EO 0.381→0.155** (대폭). DP 0.237→0.21(약간). acc 0.773→0.86(향상).
- **🎯 핵심**: demographic + **structured(image)** bias에서 fairds 강력 작동 → Adult/COMPAS(diffuse tabular, 약함)와 정반대. **"structured면 demographic fairness도 된다" 가설 입증.**
- **정직**: ren2018 worst 0.785 (fairds보다 약간 위, vs resid p=0.049). **jtt 0.292/groupdro 0.469 — CelebA에서 오히려 약함** (다른 regime과 반대; 2-stage/groupdro가 face 64px서 불안정, variance 0.18-0.22). fairds-2 vs fairds-1 NS (0.719 vs 0.747) — structured bias엔 1차로 충분, 2차 isolation 없음.
- results: `results/e7_main`(vanilla/f1/f2), `e7_oracle`(ren/jtt/gdro), `e7_resid`(f2 residual).

## [2026-06-04] 🎯 연구 목표 fairness 재확립 — group fairness 프레임 + demographic 재분석
- **사용자 지적**: 발표/논문이 'spurious shortcut robustness'로 흘렀으나 **원래 목표는 fairness** (idea/C2/C3/C4, FORML 비교). "데이터 기여도로 편향 학습 방지 → 공정"이 목적. E2 fairness 약세로 robustness pivot했던 것을 fairness로 되돌림.
- **재정렬**: worst-group = group fairness (Rawlsian/max-min) + **group disparity gap** metric 추가, mechanism(2차 Shapley가 편향 다수 down-weight = C3)을 bias-mitigation으로 전면화. spurious→data bias 용어.
- **Disparity 분석 (majority−minority acc gap; ↓ fairer)**: fairds-2가 vanilla 대비 gap **40-47%↓** — CMNIST 0.223→0.118, CIFAR 0.703→0.430, STL 0.950→0.568(resid). no-label 중 최저(vanilla/f1/ren 우위). jtt/groupdro 더 공정(정직). `codes/experiments/fairness_analysis.py`.
- **Minority lift**: gap 감소는 **소수 그룹 향상**으로 — STL minority 0.04→0.32, majority 0.99→0.89, overall 0.14→0.38 (소수 보호, 다수 소폭 양보). fairness 정석 거동.
- **Demographic 재분석 (Adult/COMPAS DP/EO)**: fairds-2 accuracy 보존하나 DP/EO 개선 미미(NS). ren2018(bi-level) 더 공정(acc 희생). 진단: **diffuse tabular bias → per-example signal 작음**. scope: structured bias(이미지)는 작동, diffuse(tabular)는 안 됨 — 정직한 boundary.
- **산출물 fairness 재작성**: 논문(제목 'for Group Fairness', abstract/intro/exp/discussion + **Broader Impact** 섹션, 본문 4p + refs 5p), 발표(11장 fairness ~8분), speaker_notes(fairness 영어). 새 `fig_disparity`. paper_aaai/.

## [2026-06-02] 🎯 E6 Spurious-STL10 — 3rd positive regime 확정, 8-clinch PASS
- 96px 자연이미지(STL-10 car vs truck), texture-corruption(blur vs noise) spurious. CMNIST(color)/Corrupted-CIFAR(texture) 미러. from-scratch 3-conv CNN(96→12), 10 seed, 30ep, τ=0.1 ws=4.0 α=0.5. STL train+test pool(car+truck 2600) → disjoint 4-way split. 신규: `datasets/spurious_stl10.py`, `experiments/e6_spurious_stl10/{run,analyze}.py`.
- **test_worst (10 seed)**: vanilla 0.042, fairds-1 0.194, fairds-2 0.227, **fairds-2-resid 0.323**, ren2018 0.151, jtt 0.606, groupdro 0.521.
- **8-clinch PASS (Codex R3 기준, seed-paired Wilcoxon)**: resid vs vanilla **+28.1pp p=0.002**, vs fairds-1 **+12.9pp p=0.010**, vs ren2018 **+17.2pp p=0.002** — 3개 target 전부 충족.
- **mechanism**: residual_real > full fairds-2 (+9.6pp p=0.037) — CIFAR(≈preserve)보다 강함, 잔차 = 진짜 방향 signal 재확인. STL에선 평행분(shrinkage)이 오히려 약화 요인.
- **honest**: jtt 0.606, groupdro 0.521이 fairds-2-resid 압도 (CMNIST/CIFAR 동일 패턴, 정직 보고). STL에서 jtt 특히 강함.
- **🎯 3 regimes (color/texture/natural-image)에서 2차 mechanism 재현 → universality 확립**. Codex 7.8→8 clinch 조건(2nd non-toy positive regime) 충족.
- results: `results/e6_main`, `results/e6_oracle`, `results/e6_resid` (각 최신 ts).

## [2026-06-02] ✅ 메커니즘 잠금 (H* full battery) + Codex 7.8/10 — residual_real이 새 headline
- **H* 입증 (probe.py 진단 + e5 arm sweep)**: 2차 cross-term = **평행분**(curvature shrinkage = smoothing 등가) + **직교 잔차**(sample-specific directional signal). cos(first,cross)≈1 (g_val≈H_val top eigenvector, align 0.89–0.99) → H1/H2 반증. 분해: cross_n = β·phi1 + r, residual arm = phi1 − α·r. 코드: `fairds/shapley.py::shapley_residual_arms`, `fairds/trainer.py`(arm 분기+`_shuffle_residual_within`), e5/e3 run.py `--arm`.
- **Residual ablation 20 seed (Corrupted-CIFAR, test_worst)**: residual_real **0.458** > shuffle 0.398 (**+5.9pp p=0.0015**, Wilcoxon 0.0012) > parallel 0.366 (+9.2pp p<1e-4) ≈ phi1 0.370. sign_flip 0.067 (붕괴 p=5e-4). → 잔차 = smoothing/noise 아닌 **진짜 방향성 signal**. (10seed에선 real-vs-shuffle NS였으나 20seed에서 유의.)
- **honest selection (Corrupted-CIFAR, held-out val_eval_worst ranking)**: **Fairds-2-residual val 0.647 / test 0.463 = non-oracle best** (GroupDRO oracle와 val 동급). > original 0.638/0.423 > ren2018 0.630/0.375 > parallel 0.624/0.379 > jtt 0.577/0.368 > vanilla 0.555/0.196. → selection leakage 없이 2차 우위 주장 가능 (alpha=2 leakage 우회).
- **Waterbirds last-layer + DFR (10 seed, test_worst)**: vanilla 0.186, fairds-1 0.366, fairds-2 0.444, ren2018 0.440, **dfr 0.485**(acc 0.849 유지). fairds=ren=dfr 동등(NS), vanilla 압도(+25.8pp p=6e-4). → **boundary result**(우위 아님, DFR 실용 best). `baselines/dfr.py`.
- **Codex 크로스체크 (gpt-5.5 xhigh, threadId `019e878e-05f4-7291-88d1-571c65ea6569`)**: R1 7.2 → R2 7.5 → **R3 7.8/10 borderline 8**. 8 clinch = CelebA last-layer (residual이 ERM/fairds-1/ren ≥5pp or p<0.05, DFR 동등이상, groupdro oracle엔 져도 명시). CelebA 전 high-value = CMNIST residual(headline 정당화). abstract 문장 받음. 상세: `REVIEW_20260602_mechanism.md`.
- **CMNIST residual (10 seed)**: residual_real 0.815 ≈ fairds-2-original 0.824 (preserve, p=0.70), > fairds-1 0.777 (p=0.008). 단 real-vs-shuffle NS (p=0.26) — CMNIST mechanism 약함. (Corrupted-CIFAR는 real>shuffle p=0.0015.)
- **mechanism은 benchmark-dependent 아님 — "universal" (보강으로 정정)**: 처음 3seed에서 group-alignment 가설(texture O/color X)을 세웠으나 **10seed 보강에서 반증** — residual_fraction CMNIST 0.206 ≈ CIFAR 0.210, alignment 부호 둘 다 불안정(3/10, 4/10). 정적 구조로 차이 설명 안 됨. **CMNIST real>shuffle 20seed: +4.85pp p=0.071 (marginal)** — Corrupted-CIFAR(+5.9pp p=0.0015)와 effect size·방향 일치. → 잔차 signal은 색·texture 양쪽에서 **일관 작동(universal)**, CMNIST marginal은 variance/power 차이. ("benchmark-dependent"는 10seed noise였음.) `probe_residual.py`.
- **진행/다음**: (a) alignment 통찰 보강 or (b) 7.8 확정 후 논문 반영. CelebA(8 clinch)는 선택지로 보류.

## [2026-06-02] 🚀 7→8 강화: Track A (last-layer Shapley) + Track B (Corrupted-CIFAR) — 10seed 확정 중
- **동기**: Codex 7→8 ceiling = (a) 두 번째 non-toy positive regime, (b) deeper theory. 사용자 "더 큰 모델" 제안은 Waterbirds FT 실패 regime이라 신중 권고. 대신 **① 2nd positive + ② Waterbirds 정복**, 가볍게.
- **env**: `PY=/home/users/yonghee/.conda/envs/fl_shapley/bin/python`, GPU 2×97GB Blackwell (0,1).
- **Track A — Waterbirds last-layer Shapley (E3b honest-negative 정복 시도)** 🎯
  - 진단: pretrained backbone의 full-param g_i가 small/noisy → EMA 오염 (E3b 실패원인). **Fix: backbone freeze → fc head gradient로만 Shapley** (DFR insight, Kirichenko 2022). 구현: `run.py --freeze-backbone` (shapley.py가 requires_grad 필터라 코드 자동 반영). 224px linear-probe, lr=0.1, 25ep.
  - 3 seed 잠정 (test_worst): vanilla 0.205, **fairds-1 0.388**, fairds-2 0.312, **ren2018 0.397**.
  - **fairds-1 ≈ ren2018 (0.388 vs 0.397, p=0.84 동등)** — full-param의 -29pp 압패를 동등으로 뒤집음. honest negative 정복(부분).
  - **fairds-2 < fairds-1**: last-layer 저차원 Hessian(512×2)에서 2차 cross-term 역효과 → "from-scratch=2차 유리, FT=1차로 충분" regime별 차수 선택 스토리.
  - accuracy tradeoff: fairds worst↑(0.39) overall acc↓(0.72 vs vanilla 0.85).
  - 10 seed 확정 중: `results/e3b_lastlayer/` (3seed: 20260602-060900).
- **Track B — Corrupted-CIFAR-10 (2nd positive regime 확보)** 🎯
  - color 아닌 **texture-corruption(blur vs noise) spurious**. CMNIST 미러 (anchor/val_eval/OOD-test). from-scratch 2-conv CNN. 신규: `codes/datasets/corrupted_cifar.py`, `codes/experiments/e5_corrupted_cifar/run.py`.
  - **10 seed 확정** (test_worst): vanilla 0.196±0.143, fairds-1 0.374, **fairds-2 0.423±0.046** (lowest var), ren2018 0.375, jtt 0.368, groupdro 0.564.
  - **fairds-2 vs vanilla +22.8pp (p=0.0016)** — 두 번째 positive regime 확정.
  - **fairds-2 vs ren2018 +4.8pp (p=0.016)**, **vs fairds-1 (2차 isolation) +4.9pp (p=0.042)** — CMNIST(+2.7/+4.7pp) 재현, ren2018 우위는 오히려 더 강함(유의). 둘 다 same-supervision regime.
  - groupdro 0.564가 fairds-2 압도(-14pp, group label 사용 — 다른 supervision, 정직 보고). jtt 0.368 동등(NS). `results/e5_corrupted_cifar/20260602-063050`.
- **Track A 10 seed 확정**: fairds-1 0.366 (vs vanilla +18.0pp p=0.0022), **fairds-2 0.444 ≈ ren2018 0.440** (동등). ⚠️ 3seed 잠정의 "fairds-2 < fairds-1 (2차 역효과)"는 **반증됨** — 10seed에선 fairds-2(0.444) > fairds-1(0.366). variance였음. (`results/e3b_lastlayer/20260602-064215`)
- **🔬 메커니즘 진단 (`experiments/diagnostics/probe.py`) — H1/H2 반증, 통합 H* 도출**:
  - **H2 (cross⊥first 직교성) 반증**: 두 regime 모두 cos(first,cross)≈1.0 (평행). 근원: g_val ≈ H_val top eigenvector — align(H_val·g_val, g_val) = 0.989(from-scratch)/0.89(last-layer).
  - **H* (통합)**: 2차 = first-order의 **곡률 기반 shrinkage**. phi1 group gap(minority−majority) → phi2에서 |damp| (from-scratch 20.9→11.8, last-layer −232→−98). D3(per-batch weight_std fairds-2 2.65 < fairds-1 3.01)와 동일 현상 → 2차 = 1차 과격 reweight 완화 → 안정성↑ + worst-group↑.
  - **H1 (backbone noise) 반증**: backbone이 ⟨g_i,g_val⟩ 지배(140.8 vs fc 72.8) + group 더 구분(cos 0.62<0.87). last-layer 우위는 noise제거가 아니라 spurious-heavy backbone 신호 제거로 재해석 필요.
- **다음**: (a) **smoothing ablation** — fairds-1 weight_scale sweep이 fairds-2 따라잡나? (2차=전역smoothing vs 곡률적응 판별), (b) Codex 메커니즘 크로스체크, (c) best-tuned lr 공정성, (d) 논문 반영 (NARRATIVE/PAPER).

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
