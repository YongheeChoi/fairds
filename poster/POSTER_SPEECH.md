# 포스터 발표 준비 노트

**논문**: Closed-form 2nd-order Data Shapley as a Spurious-Correlation Attenuator
**최종 평점**: paper 7/10 (gpt-5.5 xhigh), workshop submission-ready
**포지셔닝**: regime-dependent mechanism paper (SOTA claim 아님)

---

## 0. 한 줄 요약 (외워둘 것)

> "Wang et al.\ 2024의 closed-form In-Run Data Shapley를 *학습 중* per-sample
> reweighter로 재용도한 작업이고, 어디서 작동하고 어디서 작동 안 하는지
> 정직하게 짚는 mechanism 페이퍼."

영어:
> "We repurpose Wang et al.'s closed-form In-Run Data Shapley as an
> *in-training* per-sample reweighter. It's a regime-dependent mechanism
> paper — we pin down when it works and when bi-level methods stay better."

---

## 1. 핵심 숫자 4개 (이건 반드시 암기)

| 숫자 | 의미 | 통계 |
|---|---|---|
| **+13.0 pp** | OOD test_worst (Colored MNIST), Fairds-2 vs Vanilla | p=5×10⁻⁵, 10 seeds |
| **+4.7 pp** | 2nd-order cross-term **isolated** 이득 (Fairds-2 vs Fairds-1) | p=0.033 |
| **+55.3 pp** | spurious strength 0.99 에서 vanilla collapse vs fairds-2 robust | p=3×10⁻⁵ |
| **1.89×10⁻⁸** | mechanism direct evidence — Mann-Whitney U(maj φ < min φ) | controlled MLP |

이 4개를 외우면 통계 banner 가 없어도 본문에서 짚어가며 설명 가능.

---

## 2. 발표 흐름 (2-3분 walkthrough)

### Opening (15초)
> "잠깐 시간 괜찮으시면 60초 소개드릴게요. Wang et al.\ 2024 의 closed-form
> In-Run Data Shapley 를 학습 중 per-sample reweighter 로 재용도한 거에요.
> mechanism 페이퍼고, 어디서 작동하고 어디서 작동 안 하는지 정직하게 짚는
> 게 핵심입니다."

### Problem (30초) — 좌상단 Problem 카드 가리키며
> "Spurious correlation 환경에서 ERM 이 majority shortcut 을 잡잖아요. 표준
> 해법 셋 — bi-level meta-learning (FORML) 은 implicit differentiation
> 부담, GroupDRO/IRM 은 group label 필요, JTT 는 학습 동역학 개입 안 함.
> Wang 등이 작년에 closed-form 1차/2차 Taylor estimator 를 만들었는데
> *post-hoc* filtering 만 했어요. 우리 질문: 같은 식을 *학습 중* reweighter
> 로 쓸 수 있나? 특히 2차 cross-term 이 stand-alone benefit 을 주나?"

### Approach (30초) — 우상단 Approach 카드
> "4개 아이디어로 풀었어요. (1) 사후 필터링 대신 매 minibatch softmax
> weight, (2) 2차 cross-term 은 단순 alignment 가 아니라 *validation loss의
> 곡률이 큰 방향에 align 된 majority gradient 에 추가 페널티* — curvature-aware
> down-weighting, (3) Hessian spectrum 이 학습 중 수십 배 변동하니 cross-term
> 을 1차의 RMS 에 맞춰 재스케일 (안 하면 발산), (4) per-step φ 가 noisy
> 하니 EMA 누적."

### Method (45초) — 좌측 Method 카드
> "5 step 입니다. (1) per-sample gradient 를 `vmap(grad)` 한 그래프에서.
> (2) validation gradient 와 그것의 Hessian-vector product 를 Pearlmutter
> trick 으로. (3) 수식 φ^(2) = ⟨g_i, g_val⟩ − α·⟨g_i, H_val g_val⟩, cross-term
> 은 RMS rescale. (4) EMA 누적해서 (5) softmax weight 로 weighted SGD.
> Variants 박스 보세요 — Fairds-1 은 α=0 (1차만), Fairds-2 는 α=1 + RMS
> rescale. 두 variant 의 차이가 *cross-term contribution* 입니다."

### Phase Transition / Colored MNIST (30초) — 우측 Phase + 좌측 CMNIST
> "Phase transition 이 핵심 — spurious 강도 0.7~0.9 일 땐 vanilla 와 fairds
> 차이 없는데, 0.95 부터 vanilla 가 떨어지기 시작해서 0.99 면 거의 collapse.
> fairds-2 는 그때도 0.74 유지. fairds-2 의 진가는 spurious 가 *아주 강한*
> regime 에서 나옵니다."
>
> "Colored MNIST 표 보면 7 method 모두 있는데, 정직하게 GroupDRO 0.900,
> JTT 0.878 로 우리보다 강해요. 우리는 SOTA 가 아닙니다. 단 fairds-2 vs
> fairds-1 의 isolation gain (+4.7pp, p=0.033) 이 살아 있고, **2nd-order
> cross-term 자체의 기여를 통계적으로 분리해 보여주는 게 우리 contribution**."

### Waterbirds (30초) — 우측 Waterbirds
> "Pretrained ResNet-18 fine-tuning regime 에서 우리 method 가 무너집니다.
> 3 가지 variant (default EMA, warmup, no_ema) 다 vanilla 이하 또는 동등.
> FORML 가 명확히 dominate (Δ = -30pp). 진단은 — pretrained model 의 g_i
> 가 초기 epoch 에 small/noisy 해서 EMA 가 noise 만 누적, reweighting 이
> 학습 dynamics 망김. **이 boundary 를 정직하게 보고하고**, 이 regime 에선
> bi-level 쓰라고 권장합니다."

### Takeaway (15초) — 우하단 Takeaways
> "정리하면 — closed-form 2차 Shapley reweighting 은 **regime-dependent
> 도구** 입니다. From-scratch / spurious correlation 에서 작동, pretrained
> fine-tuning 에서 실패. Cross-term 의 RMS rescaling 이 stability 의 키.
> 페이퍼는 honest mechanism contribution 을 목표로 합니다."

---

## 3. 카드별 상세 (질문 받을 때 깊게 들어갈 내용)

### Title
- "Closed-form 2nd-order Data Shapley as a Spurious-Correlation Attenuator"
- 키워드 3개: **closed-form** (효율, single training run), **2nd-order**
  (cross-term, curvature), **spurious-correlation** (문제 영역)

### Problem We Tackle
**핵심 메시지**: spurious correlation 은 표준 문제, 기존 해법이 다 trade-off, Wang 2024 가 만든 closed-form 은 효율적이지만 post-hoc 만.

발표 시 강조할 디테일:
- Spurious correlation 예시: Colored MNIST 의 색 채널이 라벨과 상관 → 모델이 shape 대신 색 학습
- Bi-level meta (FORML): inner loop SGD + outer loop implicit differentiation. 메모리/시간 비용
- GroupDRO, IRM: group label 이 매 minibatch 필요 → 실제 데이터에서 group label 비싸거나 부재
- JTT (Just Train Twice): 2-stage ERM, ERM 한 번 학습 후 mistakes upweight. **학습 동역학 자체에 개입 안 함**
- Wang 2024 의 In-Run Data Shapley: 단일 training run 에서 closed-form 1차/2차 Taylor estimator 유도 → 효율적. 그러나 post-hoc filtering (저-Shapley 샘플 제거 후 재학습) 만 적용

핵심 gap: "효율적인 closed-form 인데 학습 중에 쓰지 않을 이유가 있나?" + "1차만 쓰는데 2차 cross-term 의 stand-alone 기여는 뭔가?"

### How We Approach It
4 아이디어 각각:

**Idea 1 — 사후 필터링 → 학습 중 가중치**
- 매 minibatch 의 per-sample φ_i 를 softmax 가중치로 환산
- 가중 SGD: ∇θ ← Σ_i w_i ∇θ L(z_i; θ_t)
- 학습 dynamics 에 *직접* 개입

**Idea 2 — 2nd-order cross-term 의 기하학적 의미**
- 1차 ⟨g_i, g_val⟩ = 단순 gradient alignment
- 2차 ⟨g_i, H_val g_val⟩ = H_val 이 큰 곡률을 가지는 방향에 g_i 가 align 된 정도
- 즉 validation loss 가 *민감한 방향* 에 majority gradient 가 가 있으면 추가 페널티
- "Validation 에서 risk 가 가장 크게 변할 방향을 majority 가 끌고 있으면 누른다"

**Idea 3 — RMS rescaling (algorithmic contribution)**
- Hessian spectrum 은 학습 초기/중기/말기에 orders of magnitude 변동
- raw cross-term 을 그대로 1차에 더하면 cross-term 의 magnitude 가 1차 압도 → 발산
- 매 step 1차 항 RMS 와 cross-term raw RMS 의 비율로 cross-term 을 rescale
- **이것 없으면 strong spurious bias 에서 fairds-2 가 chance 까지 붕괴** (ablation 으로 확인)

**Idea 4 — EMA accumulation**
- Per-step φ_i 는 minibatch noise 로 분산 큼
- iteration 간 EMA (β=0.9) 로 평활화 후 softmax-weight
- minibatch boundary 효과 완화

**추가 강조 포인트**:
- No implicit differentiation (FORML 와의 차이): single forward + backward + HVP
- Group label 은 anchor 50:50 만들 때 한 번만, 학습 *중* 에는 불필요 (vanilla validation loss 만 봄)

### Method: Fairds (5-step)

**Step 1 — Per-sample gradient g_i**
- `torch.func.vmap(grad)` 사용 — 단일 autograd graph 에서 모든 sample 의 gradient 한 번에
- naive for-loop 대비 훨씬 효율 (small CNN 에서 5-7× vanilla 비용)

**Step 2 — Validation gradient + HVP**
- g_val = anchor 의 평균 gradient
- H_val g_val 은 Pearlmutter trick (1994) 으로 — full Hessian 만들지 않고 1번의 추가 backward 로 계산
- 메모리 O(P), P = parameter 수

**Step 3 — Closed-form Shapley score**
- φ_i^(1) = ⟨g_i, g_val⟩
- φ_i^(2) = φ_i^(1) − α · \widetilde{c}_i, 여기서 \widetilde{c}_i = c_i · RMS(φ^(1)) / RMS(c)
- c_i = ⟨g_i, H_val g_val⟩

**Step 4 — EMA accumulate**
- \bar{φ}_i ← β · \bar{φ}_i + (1−β) · φ_i^(2), β=0.9

**Step 5 — Softmax weight & weighted SGD**
- w_i = clip_{q0.05, q0.95}(B · exp(\bar{φ}_i / τ) / Σ_j exp(\bar{φ}_j / τ))
- τ = 0.5
- quantile clip 으로 극단 weight 제거
- θ_{t+1} = θ_t − η · Σ_i w_i · g_i

**Variants** (포스터에 박스로 강조):
- Fairds-1: α=0 (1차만)
- Fairds-2: α=1 + RMS rescale
- **이 둘의 성능 차이 = cross-term contribution**

**Wall-clock**: 5-7× vanilla (small CNN), FORML (5.1×) 와 같은 자릿수

**Algorithmic contribution**: cross-term 의 RMS rescaling — 없으면 strong spurious bias 에서 발산. ablation 으로 검증.

### Mechanism
**목적**: cross-term 이 정말 의도한 대로 majority φ < minority φ 를 만드는지 controlled 환경에서 검증

**셋업**:
- Spurious-feature MLP (2-layer)
- 8-dim input, 2 dim 만 informative, 나머지 6 dim 은 noise + spurious correlation
- Spurious strength s=10 (강함)

**결과**:
- ratio = 0.9 에서 Mann-Whitney U test: U(majority φ < minority φ) p = **1.89×10⁻⁸**
- Δφ (= mean majority φ − mean minority φ) 가 imbalance 가 커질수록 단조 감소
  - imbalance 0.50 → +0.002
  - imbalance 0.99 → −0.025
- 즉 spurious 가 강해질수록 cross-term 이 majority 를 *더 강하게* 누른다

**해석**: cross-term 이 H_val 의 dominant direction 과 align 된 majority gradient 에 페널티 — **direct mechanism evidence**, gradient alignment 만으로는 불가능했던 결과

### Phase Transition (Spurious Strength Sweep)
**목적**: spurious 강도가 변할 때 fairds vs vanilla 차이가 어떻게 변하는지

**셋업**: Colored MNIST 에서 p_maj (majority group 의 color-label correlation) 를 sweep
- p_maj ∈ {0.7, 0.8, 0.9, 0.95, 0.99}
- 각 p 에서 vanilla vs fairds-2 비교, 10 seeds

**결과**:
- p ≤ 0.9: vanilla 와 fairds-2 차이 크지 않음 — spurious 가 약하면 vanilla 도 OK
- p = 0.95: Δ = +19.8pp 시작
- p = 0.99: Δ = **+55.3pp** (p=3×10⁻⁵) — vanilla 거의 collapse (random 수준), fairds-2 는 0.74 유지

**해석**: fairds-2 의 진가는 spurious 가 *극단적으로 강한* regime — workshop reviewer 도 이 sweep 으로 7점 도달

**중요한 정직성 포인트** (Q&A 대비): FORML 도 0.99 에서 robust (~0.68). fairds-2 (0.74) 가 FORML (0.68) 보다 약간 좋지만 차이 작음. Phase transition 은 *reweighting 일반의 효과*, fairds 만의 효과 아님.

### Colored MNIST: Held-out OOD (메인 empirical 결과)
**셋업** (Arjovsky 2019 스타일):
- 이진 분류 (digit ≥ 5)
- color channel 이 라벨과 spurious 하게 상관
- Train: 5000 examples, majority group (90%) 의 color-label correlation = 0.9, minority (10%) = 0.5
- **Anchor (학습 내부)**: 200 examples, 50:50 group-balanced
- **Val_eval (held-out, 학습 중 절대 사용 안 함)**: 500 examples, 50:50 balanced
- **OOD Test**: 5000 examples, color-label correlation 을 0.1 로 *뒤집음*. test_worst 는 spurious-flipped (color≠label) 그룹의 정확도
- Small CNN, 10 seeds, 20 epochs, best-tuned lr 0.05

**결과 표** (test_worst):
| method | test_worst |
|---|---|
| vanilla | 0.694±0.064 |
| FORML (corrected) | 0.796±0.054 |
| fairds-1 | 0.777±0.071 |
| **fairds-2** | **0.824±0.043** |
| JTT | 0.878 (외부 baseline) |
| GroupDRO | 0.900 (외부 baseline, group label 사용) |

**핵심 paired t-test**:
- fairds-2 vs vanilla: +13.0pp (p=5×10⁻⁵) ⭐
- fairds-2 vs FORML: +2.7pp (p=0.052) marginal
- fairds-2 vs fairds-1: +4.7pp (p=0.033) — **2차 isolation**

**부가 메시지**:
- fairds-2 는 seed variance 가 가장 낮음 (std 0.043 vs FORML 0.054, vanilla 0.064)
- val_eval (in-distribution) 에서는 fairds-2 ≈ FORML, OOD 에서 비로소 fairds-2 가 분리됨
- **정직 보고**: GroupDRO 0.900, JTT 0.878 이 fairds-2 (0.824) 보다 강함. SOTA claim 아님.

### Waterbirds: Regime Boundary (정직한 negative)
**셋업**:
- Pretrained ResNet-18, ImageNet 가중치, BN running stats 동결 (vmap 호환)
- n_train=4795, anchor=200 (50/group balanced), val_eval=999, test=5794
- 3 seeds × 10 epochs × image_size 96

**시도한 3 variant**:
1. Default EMA (β=0.9): 모든 hp 조합에서 catastrophic 실패
2. Warmup_epochs=3: vanilla 수준 회복 (학습 초기 EMA noise 회피)
3. No-EMA (per-step φ 사용, lr 낮춤): vanilla 보다 약간 좋음 (p=0.17 NS)

**결과** (test_worst):
| method | test_worst |
|---|---|
| vanilla | 0.182±0.033 |
| fairds-1 default | 0.134±0.095 (파멸) |
| fairds-2 default | 0.145±0.081 (파멸) |
| fairds-2 + warmup=3 | 0.200±0.086 (≈ vanilla) |
| **FORML** | **0.494–0.499** ⭐ |

FORML vs vanilla: +31.2pp (p=0.020)
fairds-2 vs FORML: −29.4pp (FORML strictly dominates)

**진단**:
- Pretrained ResNet 은 이미 competent feature extractor → 초기 epoch g_i 가 small/noisy
- EMA 버퍼가 noise 로 채워짐 → reweighting 이 학습 dynamics 망김
- Warmup 으로는 vanilla 따라잡지만 cross-term effect 활용 못 함
- No-EMA variant 도 SNR 자체가 poor 라 회복 안 됨
- Seed=2 는 모든 hp 에서 collapse (high variance), FORML 은 모든 seed 에서 stable

**Paper 가 던지는 메시지**:
- Cross-term mechanism 은 from-scratch / small-model 에서 깨끗하게 작동
- Pretrained fine-tuning 으로 *일반화되지 않는다* — 3 variant 시도해도
- 따라서 **regime-dependent mechanism paper**: closed-form in-run Shapley reweighting 이 어떤 regime 에서 작동/실패하는지 boundary 를 characterize
- 실용적 권장: fine-tuning 환경 → bi-level meta-learning (FORML)

### Takeaways
1. **Regime-dependent tool** — 일반 reweighter 가 아님
2. From-scratch / spurious-correlation regime 에서 작동, cross-term contribution 통계적 isolation (+4.7pp, p=0.033)
3. Pretrained fine-tuning regime 에서 실패, bi-level meta-learning 이 강함
4. RMS rescaling 이 stability 의 키 (없으면 발산)

---

## 4. 예상 Q&A

### Q1: JTT/GroupDRO 가 더 강하면 왜 굳이 Fairds 를 쓰나요?
**A**: SOTA claim 아닙니다. 우리는 *2차 cross-term 의 isolated 기여* 자체를
보여주는 mechanism paper. closed-form (single forward + backward + HVP) 이고
group label 은 anchor 만들 때만 필요한 점이 차별점이긴 한데, 단순히
worst-group accuracy 가 목표면 GroupDRO 쓰셔야 합니다.

### Q2: RMS rescaling 이 ad-hoc heuristic 처럼 보입니다
**A**: 맞습니다, 솔직히. 이론적 정당화는 "Hessian spectrum 이 training 중
orders of magnitude 변동" 이라는 관찰에서. ablation 으로 이게 없으면 발산하는
걸 보였고, RMS 가 working solution 인 건 empirical. 더 principled scaling
(adaptive α, spectral normalization) 이 future work.

### Q3: Phase transition 결과는 fairds-2 만의 효과인가요? FORML 도 0.99 에서 강한 것 같은데요
**A**: 정확합니다 — phase transition 은 *reweighting 일반*의 효과이지 우리
cross-term 만의 효과는 아닙니다. FORML 도 0.99 에서 0.68 정도 robust.
fairds-2 (0.74) vs FORML (0.68) 의 차이는 작아요. 우리가 unique 하게
주장하는 건 cross-term isolation 이고, phase transition 은 추가 evidence.

### Q4: Pretrained Waterbirds 실패가 method 의 큰 약점 아닌가요?
**A**: 맞습니다, 그래서 paper 에 정직하게 보고했어요. 이걸 숨기면 reviewer 가
발견하고 더 큰 문제. 대신 "closed-form Shapley 가 어떤 regime 에서 안 통하는가"
라는 question 자체가 mechanism 연구로서 가치 있습니다 — 동일한 reweighting 이
다른 regime 에서 다르게 작동하는 이유 분석.

### Q5: Hessian-vector product 비용이 큰데 scale 어떻게 하나요?
**A**: 실측: vanilla 대비 5-7× wall-clock (small CNN). FORML (5.1×) 와 같은
자릿수입니다. Foundation model scale 에선 Wang et al.\ 의 ghost technique 으로
더 줄일 수 있다고 봅니다. 단 이 페이퍼에선 효율 주장은 보수적으로, "comparable
to bi-level" 정도만 합니다.

### Q6: 왜 group label 이 anchor 에서만 필요한가요?
**A**: Anchor 를 group-balanced 50:50 으로 만들 때 한 번 사용. 학습 중에는
anchor 의 vanilla validation loss 만 봅니다. ARL (Lahoti 2020) 같은 방법은
anchor 도 group label 없이 만들 수 있으니, 우리 방법은 *완전히*
group-label-free 는 아닙니다 (paper 에 명시). FORML 도 비슷한 anchor 가정.

### Q7: Cross-term mechanism 의 이론적 보장은?
**A**: Wang et al.\ 2024 의 Taylor expansion 유도가 있고 우리는 그걸
in-training 으로 가져왔을 뿐. cross-term 의 "majority down-weighting" 효과는
controlled MLP 의 empirical 입증 (p=1.89×10⁻⁸) 으로 보였지만, formal 한
theoretical guarantee 는 없습니다. 이게 paper 가 6/10→7/10 ceiling 인 이유
중 하나.

### Q8: EMA decay β=0.9 가 어떻게 정해졌나요?
**A**: Ablation. β=0.9 가 balance. 너무 작으면 (0.1) per-step noise 그대로,
너무 크면 (0.99) 너무 stale 해서 학습 변화 따라가지 못함. β, α, τ 모두
ablation 으로 fix.

### Q9: 1차 항만으로 의미 있나요? 왜 2차가 필요한가요?
**A**: Fairds-1 (1차만) 도 vanilla 대비 +8pp 정도 향상 (test_worst 0.777 vs
vanilla 0.694). 즉 1차 reweighting 자체로도 일부 효과. 다만 fairds-2 의
+4.7pp 추가 isolation 이 *cross-term 의 stand-alone* 기여. mechanism 카드의
Mann-Whitney p=1.89×10⁻⁸ 이 그 mechanism 의 direct evidence.

### Q10: Anchor size 200 이 너무 작지 않나요? Hessian 추정 noisy 할 텐데
**A**: Anchor 는 group-balanced 50:50 으로 만들기에 200 (각 group 100) 으로
충분. H_val g_val 의 Pearlmutter HVP 는 anchor 의 *평균* gradient 의 Hessian-vector
product 라 individual sample noise 가 평균화됨. 단 큰 모델로 가면 anchor
size 조정이 필요할 수 있고 그건 future work.

---

## 5. 발표 시 주의 점

- **"SOTA claim 아니다"** 를 명확히. GroupDRO/JTT 더 강하다고 먼저 말하면
  reviewer/청중이 안심.
- **Waterbirds 실패** 를 먼저 인정. "정직하게 봤고, regime boundary 를
  characterize 하는 게 contribution" 으로 framing.
- **Cross-term isolation** 이 진짜 핵심. +13pp vs vanilla 보다 +4.7pp vs
  fairds-1 (2차 isolation) 이 우리가 우선적으로 주장하는 것.
- **Mechanism evidence** (Mann-Whitney p=1.89e-8) 가 다른 reweighting paper
  들과 차별화되는 지점.

## 6. 만약 시간 부족하면 (60초 버전)

1. Closed-form In-Run Shapley (Wang 2024) 를 학습 *중* reweighter 로 재용도
2. 2차 cross-term 의 stand-alone benefit 을 통계적으로 isolate (+4.7pp, p=0.033)
3. RMS rescaling 이 algorithmic contribution (없으면 발산)
4. 작동 regime: from-scratch / small-model / spurious — Colored MNIST 에서 +13pp vs vanilla
5. 실패 regime: pretrained ResNet fine-tuning — 정직 보고, bi-level 권장
