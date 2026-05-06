# Research Plan: 동적 In-Run Data Shapley를 활용한 단일 학습 기반 적응형 편향 완화 알고리즘 (Fairds)

> 이 파일은 `/research-wiki ingest-plan <plan.md>` 입력 형식에 맞춰 작성됨. 섹션 헤더와 stable id (`C1`, `G1`, `E1`) 는 자동 추출 안정성을 위해 변경하지 말 것.
> 본 wiki 의 디렉토리·페이지 구성은 [CLAUDE.md](CLAUDE.md), 카탈로그는 [index.md](index.md) 참조.

## Background / Literature

이 연구가 토대로 삼는 핵심 선행 연구. arxiv ID 가 확인되지 않은 항목은 venue·저자 정보로 대체 (수동 메타데이터 보완 필요).

- **Data Shapley in One Training Run (In-Run Data Shapley)** (venue: ICLR 2025 Outstanding Paper Runner-Up; authors: Wang, J. T. et al., 2024) — Taylor 1차/2차 근사 + ghost technique 으로 단일 학습 과정 내에서 매 gradient update 마다 Shapley value 를 closed-form 으로 계산. **본 연구의 직접 기반 기술.** 원 논문은 사후 (post-hoc) 데이터 필터링용. → [[wiki/papers/in-run-data-shapley]]
- **FORML: Fairness Optimized Reweighting via Meta-Learning** (venue: ICML 2022 DataPerf Workshop; authors: Yan et al., 2022, Apple Research) — bi-level meta-learning 으로 소규모 exemplar set 의 공정성 손실을 outer 목적으로 두어 동적 reweight. **가장 유사한 비교군이자 차별화 핵심 대상.** → [[wiki/papers/forml]]
- **FairShap** (authors: Arnaiz-Rodriguez et al., 2023) — Shapley value 기반 공정성 가중치, 단 학습 전 1 회 계산 후 고정 (정적 전처리). → [[wiki/papers/fairshap]]
- **Learning to Reweight Examples for Robust Deep Learning** (venue: ICML 2018; authors: Ren, M., Zeng, W., Yang, B., Urtasun, R.) — 검증 셋 기반 메타러닝 reweight 의 원조. 1-step bi-level approximation. → [[wiki/papers/ren2018]]
- **Fairness without Demographics through Adversarially Reweighted Learning (ARL)** (venue: NeurIPS 2020; authors: Lahoti, P. et al.) — adversarial reweighting 으로 민감속성 레이블 없이 약자 그룹 자동 식별. "민감속성 불요" 측면에서 본 연구와 목표 공유. → [[wiki/papers/arl]]
- **CHG Shapley** (authors: Cai, 2024) — 동시기 single-run Shapley 변형, 정확도 (sample efficiency) 최적화 초점. 메타데이터 미확인 (후속 보완 필요). → [[wiki/papers/chg-shapley]]

## Research Question

> **In-Run Data Shapley 의 closed-form Taylor 근사를 학습 중 실시간 reweighting 으로 전용했을 때, 메타러닝 기반 bi-level optimization (FORML 등) 대비 (a) 동등 이상의 공정성 성능, (b) 더 낮은 연산 오버헤드, (c) 더 약한 민감속성 레이블 의존을 동시에 달성할 수 있는가? 또한 2 차 근사의 gradient-Hessian-gradient cross-term 이 명시적 그룹 제약 없이 representation bias 를 자동 완화하는 메커니즘으로 작동하는가?**

## Hypotheses

검증 가능한 claim. 각 항목은 [[wiki/synthesis/open-questions]] 의 해당 Q 와 대응됨.

- **C1**: 1 차 Fairds 의 per-iteration 연산 오버헤드는 표준 학습 대비 **≤ 1.5×**, 2 차 Fairds 는 **≤ 3.0×** wall-clock time 으로 측정된다 (FORML 의 bi-level overhead 보다 작거나 비슷). [Q4]
- **C2**: Adult / COMPAS / CelebA 벤치마크에서 Fairds 는 FORML 대비 **DP-diff 와 EO-diff 양쪽에서 동등 이상 (≤)**, accuracy 는 동등 이상 (≥) 을 달성한다 (head-to-head, 동일 검증셋·동일 random seed). [Q3]
- **C3**: 2 차 Fairds 는 다수 그룹 데이터의 평균 Shapley value 가 소수 그룹 대비 **유의하게 낮음** (95% CI 비중첩) 을 보인다. 2 그룹 90:10 toy logistic regression 에서 시작하여 실데이터까지 확장 검증. [Q1] 🔴 **가장 결정적**
- **C4**: 검증 데이터에 민감속성 레이블이 **전혀 없는 무작위 균형 샘플** 만으로도 Fairds 는 fairness 지표 회복률을 레이블 있는 조건 대비 **≥ 90%** 로 유지한다. (FORML 은 이 조건에서 작동 불가 → 자동 우위) [Q2, Q3 ablation]
- **C5**: Fairds 의 가중치는 Shapley 4 공리를 근사 오차 한도 내에서 보존한다. 구체적으로 **(i) Symmetry 위반율 < 5%**, **(ii) Efficiency 오차 < 10%**, **(iii) Dummy 위반 사례 0**. [Q6, [[wiki/concepts/meta-fairness]]]
- **C6**: $D_{val}$ 의 크기·구성에 대한 sensitivity 는 제한적임 — $|D_{val}| \in \{500, 2000, 10000\}$ 와 그룹 비율 ±20% 변화에서 fairness 지표 변동이 **≤ 2 %p**. [Q2]

## Identified Gaps

이 연구가 메우려는 field gap.

- **G1**: [[wiki/papers/in-run-data-shapley|In-Run Data Shapley]] 는 학습 종료 후 저품질 데이터 필터링용 (post-hoc) 으로만 활용되었으며, **공정성 목적의 학습 중 실시간 (in-run) reweighting 으로 적용된 바 없다.**
- **G2**: [[wiki/papers/forml|FORML]] 등 메타러닝 reweight 는 가중치를 gradient descent 로 *학습* 하므로 **가중치 할당 자체의 공정성 (meta-fairness) 에 대한 이론적 보장이 없다** (동등한 기여를 한 두 데이터가 다른 가중치를 받을 수 있음).
- **G3**: FORML 등은 exemplar/검증 셋에서 **그룹별 공정성 지표 계산을 위해 민감속성 레이블이 필수** — GDPR 등 법적 제약 및 비용 측면에서 실용성 한계.
- **G4**: 메타러닝 reweight 의 실증은 **CIFAR / CelebA 등 소규모 분류 과제** 에 한정 — foundation model 규모에서의 실시간 편향 완화는 미검증.
- **G5**: 2 차 Shapley 의 gradient-Hessian-gradient cross-term 이 **다수 그룹의 정보 중복으로 인해 representation bias 를 자동 완화** 할 가능성에 대한 이론적·실험적 분석 부재. ([[wiki/papers/in-run-data-shapley|원 논문]] 은 이 항을 효율적 계산 기법의 부산물로만 다룸.)

## Proposed Idea

> **slug: dynamic-in-run-shapley-fairness**

[[wiki/papers/in-run-data-shapley|In-Run Data Shapley]] (Wang et al., 2024) 의 closed-form Taylor 근사 (1 차: gradient dot-product, 2 차: gradient-Hessian-gradient product) 를 사후 데이터 평가 도구가 아니라 **학습 중 편향을 실시간 교정하는 동적 제어기 (Dynamic Controller)** 로 재용도한다 ([G1]). 인구통계학적으로 균형 잡힌 소규모 [[wiki/concepts/unbiased-validation-set|unbiased validation set]] $D_{val}$ 을 anchor 로 두고, 매 iteration $t$ 에서 훈련 샘플 $z_i$ 의 Shapley value 를 $\phi_i^{(1)} \propto \langle \nabla L(z_i; \theta_t), \nabla L(D_{val}; \theta_t) \rangle$ (또는 2 차 cross-term 보정) 로 산출 → EMA 누적 → softmax/clip 으로 가중치 $w_i$ 변환 → weighted SGD 업데이트 ([[wiki/methods/fairds-algorithm|5-step pipeline]]). [[wiki/concepts/ghost-dot-product|Ghost technique]] 덕분에 추가 1-2 회 backward pass 만으로 미니배치 전체에 동시 산출이 가능하여 실시간 동작이 성립한다.

이 방식은 [[wiki/papers/forml|FORML]] 의 bi-level meta-learning 을 **closed-form Shapley 도출** 로 대체함으로써 implicit differentiation 비용을 0 으로 만들고 ([G1, C1]), 가중치가 [[wiki/concepts/shapley-value|Shapley 4 공리]] 를 만족하므로 [[wiki/concepts/meta-fairness|meta-fairness]] 라는 새로운 보장 차원을 도입한다 ([G2, C5]). 효용 함수가 단순 validation loss 이므로 검증 데이터에 민감속성 레이블이 불필요하며 ([G3, C4]), 2 차 cross-term 이 다수 그룹 데이터들 간의 정보 중복을 통해 [[wiki/concepts/representation-bias|representation bias]] 를 자동으로 감쇠시킨다는 것이 가장 독창적이지만 가장 미증명된 가설이다 ([G5, C3]). 기반 기술이 GPT-2 / Pythia-410M 까지 검증되었으므로 foundation model 규모로의 확장 가능성도 직접 상속한다 ([G4]).

## Planned Experiments

각 실험은 [[wiki/synthesis/experimental-plan|5-Phase 권고]] 의 항목과 대응. cost-effectiveness 순으로 우선순위 부여.

- **E1**: *Toy 2-group logistic regression* — tests **C3**. Dataset: synthetic Gaussian mixture (그룹 비율 50:50, 70:30, 90:10, 99:1; 5 seed). Baseline: 1 차 Shapley only, ablation 으로 2 차 추가 효과 측정. Success: 90:10 조건에서 2 차 Shapley 의 **다수 그룹 평균 < 소수 그룹 평균 (Wilcoxon p < 0.05)**, 그룹 비율 unbalance 가 커질수록 격차 단조 증가. → Phase 1, 즉시 착수.
- **E2**: *Adult / COMPAS head-to-head* — tests **C2, C4**. Dataset: UCI Adult, ProPublica COMPAS. Model: 2-layer MLP (Adult), logistic regression (COMPAS). Baselines: Vanilla, FairShap, FORML, ARL, Fairds-1차, Fairds-2차. Metrics: Accuracy, DP-diff, EO-diff. **Validation 셋 민감속성 ablation**: 레이블 있음 vs 없음 양 조건. Success: Fairds 가 FORML 대비 두 fairness 지표 모두에서 동등 이상 (한쪽 ≥ 1 %p 우위), accuracy drop ≤ 2 %p, **민감속성 없음 조건에서 fairness 회복률 ≥ 90% 유지**. → Phase 2.
- **E3**: *CelebA 얼굴 속성 예측* — tests **C2**. Model: ResNet-50. Sensitive attr: 성별. Baselines / metrics 동일. Success: FORML 의 EO-diff 개선폭 (~5%) 이상 달성. → Phase 2.
- **E4**: *Wall-clock overhead 측정* — tests **C1**. Vanilla / Fairds-1차 / Fairds-2차 / FORML 의 GPU time per epoch 비교 (동일 hardware, ResNet-18 on CIFAR-10, batch 256). Success: 1 차 ≤ 1.5×, 2 차 ≤ 3.0×, FORML 보다 작거나 비슷. → Phase 4 와 병행.
- **E5**: *$D_{val}$ sensitivity* — tests **C4, C6**. Adult/COMPAS 에서 $|D_{val}| \in \{100, 500, 2000, 10000\}$ × 그룹 비율 $\{50:50, 70:30, 인구비, 30:70\}$. Adversarial perturbation 추가 (의도적 편향 주입 시 모델이 그 방향으로 학습되는지). Success: 정상 범위에서 fairness 변동 ≤ 2 %p, adversarial 조건에서 anchor 결정성 입증. → Phase 3.
- **E6**: *Shapley 공리 보존 측정* — tests **C5**. 인공 controlled scenario 에서 동일 데이터 복제, 더미 추가 등 → Symmetry 위반율, Dummy 위반 횟수, Efficiency 오차 (clipping 전후) 측정. Baseline: FORML 가중치는 같은 기준으로 위반율 산출 가능. Success: C5 의 (i)(ii)(iii) 임계 만족, FORML 보다 명백히 낮은 위반율. → Phase 5.
- **E7**: *Small LM 사전학습 (GPT-2 small)* — tests **C2 at scale**. WikiText 또는 The Pile subset 사전학습 + BBQ / BOLD bias benchmark 평가. Baselines: Vanilla 사전학습, Fairds-1차 (2 차는 비용 부담 시 제외). Success: Fairds 가 BBQ / BOLD bias score 를 의미 있게 (effect size ≥ small) 감소시키며 perplexity 저하 ≤ 1 %. → Phase 4, 자원 확보 후.

## Success Criteria

> **Main contribution 성립 조건**: C2 (FORML 대비 동등 이상 fairness) **AND** C3 (2 차 자동 균형 효과 입증) 가 모두 *supported* 일 때. 두 가지가 효율성 차별점 (C1) 과 결합되면 "closed-form Shapley 가 학습된 가중치를 대체할 수 있다" 는 핵심 주장 성립.
>
> **Strong contribution**: 위에 더해 C4 (민감속성 레이블 불요) 가 supported → FORML 이 본질적으로 풀 수 없는 시나리오에서의 우위 확립 → 단순 incremental 이 아닌 paradigm-level 차별화.
>
> **Theoretical contribution**: C5 가 supported → meta-fairness 를 첫 도입한 가중치 reweight 방법으로 자리매김.
>
> **Failure modes**: C3 *unsupported* 시 → 2 차 차별점 [[wiki/synthesis/open-questions|차별점 ④]] 를 철회하고 1 차 only paper 로 후퇴 (incremental 평가 위험). C2 *unsupported* 시 → 이론 차별만으로는 ML 학회 통과 어려움 → 실험 설계 또는 알고리즘 재검토 필요.

---

## 부록: stable id ↔ wiki node 대응

| Plan id | wiki page (해당 / 검증 위치) |
|---|---|
| C1 | [[wiki/synthesis/open-questions]] Q4, [[wiki/methods/fairds-algorithm]] §오버헤드 분석 |
| C2 | [[wiki/comparisons/forml-vs-fairds]], [[wiki/synthesis/experimental-plan]] Phase 2 |
| C3 | [[wiki/methods/second-order-shapley]] §자동 균형 메커니즘, [[wiki/concepts/representation-bias]], [[wiki/synthesis/open-questions]] Q1 |
| C4 | [[wiki/concepts/unbiased-validation-set]], [[wiki/comparisons/forml-vs-fairds]] §차별점 ③ |
| C5 | [[wiki/concepts/meta-fairness]], [[wiki/concepts/shapley-value]] |
| C6 | [[wiki/concepts/unbiased-validation-set]], [[wiki/synthesis/open-questions]] Q2 |
| G1 | [[wiki/papers/in-run-data-shapley]] §본 연구와의 관계 |
| G2 | [[wiki/concepts/meta-fairness]], [[wiki/comparisons/comprehensive-table]] §이론적 보장 |
| G3 | [[wiki/comparisons/forml-vs-fairds]] §차별점 ③ |
| G4 | [[wiki/comparisons/comprehensive-table]] §검증된 모델 규모 |
| G5 | [[wiki/methods/second-order-shapley]], [[wiki/synthesis/open-questions]] Q1 |
| E1-E7 | [[wiki/synthesis/experimental-plan]] Phase 1-5 와 1:1 대응 |
