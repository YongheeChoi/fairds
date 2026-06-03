# Codex 크로스체크 리뷰 — 메커니즘 & model selection (2026-06-02)

- **Reviewer**: gpt-5.5 (xhigh, Codex MCP)
- **threadId**: `019e878e-05f4-7291-88d1-571c65ea6569` (resume 가능)
- **Verdict**: **7.2/10** (정직히 쓰면 7.5, 8은 아직)

## 한 줄 요약
Corrupted-CIFAR = real improvement. Waterbirds last-layer는 catastrophic은 없앴지만 DFR-style last-layer retraining처럼 보임(broadly robust 새 원리 아님). **alpha-selection이 가장 큰 위험.**

## Q1 — 메커니즘 H* (curvature residual)
- **강한 공격**: `H_val·g_val ≈ λ·g_val` → `cross ≈ λ·phi1` → RMS-norm 후 `phi2 ≈ (1−α)·phi1`. α=0.5는 smoothing, α>1은 sign inversion. alpha monotone sweep + saturation이 *곡률 잔차 없이도* 설명됨. smoothing ablation도 약함(+3.2pp p=0.17).
- **결론**: H*는 **hypothesis로만** phrasing. "not mere smoothing" 주장 금지 (residual 증명 전).
- **결정적 실험 (project-out + permutation control)**: 배치마다 `β=<cross_n,phi1>/||phi1||²`, `r=cross_n−β·phi1`. weight entropy/ESS 맞춘 5 arm:
  1. phi1 (best tau)
  2. parallel-only: `(1−α·β)·phi1`
  3. residual-real: `phi1 − γ·r`
  4. residual-shuffled: `phi1 − γ·shuffle(r)` (class + phi1-quantile bin 내 셔플)
  5. (옵션) sign-flip: `phi1 + γ·r`
  - **H*는 real > shuffled AND parallel-only일 때만 생존.** 아니면 정체는 smoothing/sign-inversion.

## Q2 — model selection 딜레마
- fatal 아님. 단 **test-selected α≥2(0.489)는 diagnostic/leakage upper bound로만 보고.**
- **표준 protocol**: 별도 group-labeled `D_select`로 worst-group acc 기준 선택 (JTT, GroupDRO, WILDS, DFR 모두). 분리: `D_train / D_anchor(학습 내부) / D_select(group-labeled, 학습 미사용) / D_test(1회)`.
- group-labeled val 불가 시: α를 dev datasets에서 globally fix, 또는 training-only rule(target weight entropy).
- **winner 바뀜**: 정직 선택 → Corrupted-CIFAR α=0.25 → test 0.433 (vanilla/Ren/fairds-1 이김, **GroupDRO 0.564엔 짐**). Waterbirds last-layer = Ren2018 동등(안 이김). **DFR baseline 필수**.
- 인용: Sagawa 1911.08731, Liu(JTT) 2107.09044, WILDS, Kirichenko(DFR) 2204.02937, LaBonte 2309.08534.

## Q3 — score & 최소 패키지
- 7.0 → **7.2** (정직히 쓰면 7.5). 약점 = **selection + mechanism entanglement**.
- **패키지 (acceptance-lift per GPU-hour 순)**:
  1. **Residual ablation** (Corrupted-CIFAR, 10 seed) — 곡률 story 진위 결정.
  2. **Honest model-selection rerun** — 전 method 동일 grid, D_anchor≠D_select, selected α 보고(test-best 아님).
  3. **Waterbirds last-layer + DFR baseline** — 동일 frozen backbone/data budget. fairds가 DFR/Ren 동등이면 그렇게 명시.
  4. (여유) **CelebA frozen last-layer**, 5 seed (ERM/Ren/DFR/fairds-1·2/GroupDRO) — last-layer 결과가 one-off 아님 입증.
- **acceptance path**: residual-real > shuffled + 정직선택서 fairds-2 > Ren + CelebA/Waterbirds 경쟁력 → **8 근접**. residual 실패 → "closed-form adaptive reweighting often works" (2차 Shapley mechanism paper 아님).
