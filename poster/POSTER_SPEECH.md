# Poster Presentation Script

**Paper**: Closed-form 2nd-order Data Shapley as a Spurious-Correlation Attenuator
**Venue**: ICLR 2026 Workshop
**Estimated time**: 2–3 minutes (quick walkthrough)

---

## Opening (15 seconds)

> "안녕하세요, 잠깐 시간 괜찮으시면 빠르게 소개드릴게요. Wang et al. 2024 의 closed-form In-Run Data Shapley 를 학습 *중* per-sample reweighter 로 재용도한 작업이에요. 메커니즘 페이퍼고, 어디서 작동하고 어디서 작동 안 하는지 정직하게 짚는 게 핵심입니다."

(영어):
> "Hi! Quick 60-second overview if you have a minute. We re-purpose Wang et al.'s closed-form In-Run Data Shapley as an *in-training* per-sample reweighter. It's a mechanism paper — we pin down when it works and when bi-level methods stay better."

---

## Motivation (30 seconds) — point at "Background" card (top-left)

> "Spurious correlation 환경에서 ERM 모델이 majority shortcut 을 잡잖아요. 표준 해법은 Ren 2018 / FORML 같은 bi-level meta-learning 인데, implicit differentiation 부담이 큽니다. Wang 등이 작년에 closed-form 1차/2차 Taylor estimator 를 만들었는데 *post-hoc* filtering 만 했어요. 우리 질문: 같은 식을 *학습 중* reweighter 로 쓸 수 있을까? 특히 2차 cross-term 이 stand-alone benefit 을 주나?"

---

## Stats Banner (15 seconds) — point at top stat callouts

> "결론 먼저 — Colored MNIST 에서 OOD test_worst 기준 Vanilla 대비 +13pp ($p=5\times10^{-5}$, 10 seeds), 1차만 쓸 때보다 2차 cross-term 의 isolated gain 이 +4.7pp ($p=0.033$). spurious 강도가 0.99 까지 가면 vanilla 가 무너지는데 fairds-2 는 여전히 robust — Δ가 +55pp 까지 벌어집니다. controlled MLP 에서 Mann-Whitney $p=1.89\times10^{-8}$ 로 메커니즘도 확인했어요."

---

## Method (45 seconds) — point at "Method: Fairds" card (col 2)

> "5 step pipeline 입니다. (1) per-sample gradient $g_i$ 를 `torch.func.vmap(grad)` 로 한 그래프에 계산. (2) validation gradient 와 그것의 Hessian-vector product 를 Pearlmutter trick 으로. (3) 수식이 $\phi_i^{(2)} = \langle g_i, g_{val}\rangle - \alpha \langle g_i, H_{val}\, g_{val}\rangle$. (4) EMA 누적해서 (5) softmax weight 으로 weighted SGD. \
> \
> **Algorithmic contribution 은 cross-term 의 RMS rescaling** 이에요. 안 하면 strong spurious 에서 chance 까지 떨어집니다. Hessian spectrum 이 학습 중에 orders of magnitude 로 변해서요."

---

## Key Results (30 seconds) — point at Results column (col 3)

> "Colored MNIST extended baseline 결과 — 여기 표가 7 method 모두입니다. 정직하게 말하면 GroupDRO 가 0.900, JTT 가 0.878 로 우리보다 강해요. 우리는 SOTA 가 아닙니다. 단 fairds-2 vs fairds-1 의 isolation gain (p=0.033) 이 살아 있고, **2nd-order cross-term 자체의 기여를 통계적으로 분리해 보여주는 게 우리 contribution** 입니다."

> "그리고 **phase transition** 이 핵심 — 옆 그림 보세요. spurious 강도가 0.7-0.8 일 땐 vanilla 와 fairds 차이 없는데, 0.95 부터 vanilla 가 떨어지기 시작해서 0.99 면 거의 collapse. fairds-2 는 그때도 0.74 유지. 즉 **fairds-2 의 진가는 spurious 가 아주 강한 regime** 에서 나옵니다."

---

## Negative Result / Honest Boundary (30 seconds) — point at "Waterbirds" card (col 4)

> "Pretrained ResNet-18 fine-tuning regime 에서 우리 method 가 무너집니다. 3 가지 algorithmic variant 다 시도했는데 (default EMA, warmup, no_ema per-step) 다 vanilla 이하 또는 vanilla 동등. ren2018 가 명확히 dominate 합니다 (Δ = -30pp). \
> \
> 진단은 — pretrained model 의 $g_i$ 가 초기 epoch 에서 small / noisy 해서 EMA 가 noise 만 누적하고, 결과적으로 reweighting 이 학습 dynamics 망김. **우리는 이 boundary 를 정직하게 보고하고**, 이 regime 에선 bi-level 쓰라고 권장합니다."

---

## Takeaway (15 seconds) — point at Takeaways highlight card

> "정리하면 — closed-form 2nd-order Shapley reweighting 은 **regime-dependent 도구** 입니다. From-scratch / spurious correlation 에서 작동, pretrained fine-tuning 에서 실패. Cross-term 의 RMS rescaling 이 stability 의 키. 페이퍼는 honest mechanism contribution 을 목표로 합니다."

---

## Closing

> "코드는 이 QR / GitHub 링크에 다 있어요. 질문 있으시면 편하게 받겠습니다!"

---

# Anticipated Q&A

### Q1: "JTT/GroupDRO 가 더 강하면 왜 굳이 Fairds 를 쓰나요?"

**A:** "솔직히, 단순히 worst-group accuracy 를 올리는 게 목표면 GroupDRO 쓰셔야 합니다. 우리는 *2차 cross-term 의 isolated 기여* 자체를 보여주는 mechanism paper 입니다. 또 closed-form (single forward + backward + HVP) 이고 group label 도 anchor 만들 때만 필요한 점이 차별점이긴 한데 — 정직히 SOTA claim 은 아닙니다."

### Q2: "RMS rescaling 이 ad-hoc heuristic 처럼 보입니다"

**A:** "맞습니다, 솔직히. 이론적 정당화는 'Hessian spectrum 이 training 중 orders of magnitude 변동' 이라는 관찰에서 옵니다. 다만 이게 없으면 발산하는 걸 ablation 으로 보였고, RMS 가 working solution 이라는 건 empirical 입니다. 더 principled 한 scaling — adaptive α, 또는 spectral normalization — 이 future work 입니다."

### Q3: "Phase transition 결과는 fairds-2 만의 효과인가요? Ren2018 도 0.99 에서 강한 것 같은데요"

**A:** "정확합니다 — phase transition 은 *reweighting 일반* 의 효과이지 우리 cross-term 만의 효과는 아닙니다. ren2018 도 0.99 에서 0.68 정도 robust. fairds-2 (0.74) vs ren2018 (0.68) 의 차이는 작아요. 우리가 unique 하게 주장하는 건 cross-term isolation 이고, phase transition 은 추가 evidence 입니다."

### Q4: "pretrained Waterbirds 실패가 method 의 큰 약점 아닌가요?"

**A:** "맞습니다, 그래서 우리가 paper 에 정직하게 보고했어요. 이걸 숨기면 reviewer 가 발견하고 더 큰 문제. 대신 \"closed-form Shapley 가 어떤 regime 에서 안 통하는가\" 라는 question 자체가 mechanism 연구로서 가치 있습니다 — 동일한 reweighting 이 다른 regime 에서 다르게 작동하는 이유 분석."

### Q5: "Hessian-vector product 비용이 큰데 scale 어떻게 하나요?"

**A:** "실측: vanilla 대비 8.2× wall-clock (CIFAR-10 ResNet-18). Ren2018 (5.1×) 와 같은 자릿수입니다. Foundation model scale 에선 Wang et al. 의 ghost technique 으로 더 줄일 수 있다고 봅니다. 단 이 페이퍼에선 효율 주장은 보수적으로, 'comparable to bi-level' 정도만 합니다."

### Q6: "왜 group label 이 anchor 에서만 필요한가요?"

**A:** "anchor 를 group-balanced 50:50 으로 만들 때 한 번 사용. 학습 중에는 anchor 의 vanilla validation loss 만 봅니다. ARL (Lahoti 2020) 같은 방법은 anchor 도 group label 없이 만들 수 있으니, 우리 방법은 *완전히* group-label-free 는 아닙니다 (paper 에 명시). FORML 도 비슷한 anchor 를 가정합니다."

---

## Print checklist

- [ ] A0 landscape (1189 × 841 mm) 인쇄
- [ ] 300 DPI 또는 그 이상
- [ ] PDF 직접 인쇄 (PPTX 가 아닌 main.pdf 사용)
- [ ] Color profile: CMYK 변환 시 ICLR primary green (#065F46) 색감 보존 확인
- [ ] QR code 추가 (paper PDF + code repo URL)
