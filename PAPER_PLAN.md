# Paper Plan

**Title (working)**: *Closed-form 2nd-order Data Shapley as a Spurious-Correlation Attenuator: Where it Works and Where it Doesn't*

**One-sentence contribution**: Repurposing Wang et al.'s closed-form In-Run Data Shapley with an RMS-rescaled 2nd-order gradient–Hessian–gradient cross-term as an in-training reweighter improves OOD worst-group accuracy on Colored MNIST by +13 pp (p=5e-5) and isolates a +4.7 pp 2nd-order contribution (p=0.033), but fails on pretrained-ResNet Waterbirds across three algorithmic variants — characterizing a regime boundary for closed-form Shapley reweighting.

**Venue**: ICLR_workshop (4-5 page main body, references not in page count)

**Type**: Empirical / diagnostic — *positive in one regime, negative in another, with a mechanism explanation.*

**Date**: 2026-05-07

**Page budget**: 5 pages main body (workshop submission). +references +appendix.

**Section count**: 6 (Intro, Related, Method, Experiments, Discussion, Conclusion).

---

## Claims-Evidence Matrix

| ID | Claim | Evidence | Status | Section |
|----|-------|----------|--------|---------|
| **C1** | Closed-form 2nd-order Shapley reweighting (Fairds-2) improves OOD worst-group accuracy on a standard spurious-correlation benchmark vs vanilla. | Colored MNIST: test_worst Δ=+13.0 pp, paired p=5e-5 (10 seeds, best-tuned lr=0.05) | **Supported** | §4.1 |
| **C2** | 2nd-order cross-term contributes a statistically distinct gain over 1st-order alone. | Colored MNIST: fairds-2 − fairds-1 = +4.7 pp test_worst, p=0.033 | **Supported** | §4.1 |
| **C3** | Fairds-2 marginally beats best-tuned 1-step bi-level meta-learner (Ren et al. 2018, FORML proxy) on Colored MNIST. | Δ=+2.7 pp test_worst, p=0.052 (one-sided paired) | **Marginal** | §4.1 |
| **C4** | Mechanism: cross-term penalizes majority samples whose gradients align with the validation Hessian's dominant directions. | E1b spurious-feature MLP: Mann-Whitney U(maj<min φ) p=1.89e-08 at imbalance 0.9 | **Supported** | §4.2 |
| **C5** | RMS rescaling of the cross-term is required to prevent divergence under strong spurious bias. | Ablation: without RMS, fairds-2 → chance acc (E1b) | **Supported** | §3 / Appendix |
| **C6** | Fairds fails on pretrained-ResNet fine-tuning regime (Waterbirds), under three algorithmic variants. | Waterbirds default / warmup_epochs=3 / no_ema per-step: all fairds-2 ≤ vanilla, far below corrected Ren2018 (Δ ≈ −30 pp test_worst, p=0.997) | **Negative finding (honest)** | §4.3 |
| **C7** | Regime boundary characterized: from-scratch / small-model spurious works; pretrained fine-tuning fails. Practical recommendation: use bi-level for the latter. | Combined evidence + diagnosis (EMA noise + small fine-tuning gradients) | **Discussion** | §5 |

---

## Structure (6 sections, ~5 pages)

### §0 Abstract (≤200 words)

- **What we achieve**: Closed-form 2nd-order Data Shapley reweighting that improves OOD worst-group acc by +13 pp on Colored MNIST.
- **Why it matters / is hard**: Bi-level meta-learning (Ren2018, FORML) is the current go-to but requires implicit differentiation. Wang et al. 2024 showed Shapley can be computed in one training run, but applied it post-hoc for filtering — never as an in-training fairness controller.
- **How we do it**: Recast In-Run Data Shapley as a per-sample weighter; introduce RMS-rescaled 2nd-order cross-term (otherwise diverges).
- **Evidence**: Colored MNIST +13 pp test_worst (p=5e-5, 10 seeds), +4.7 pp 2nd-order isolation (p=0.033). Mechanism verified with Mann-Whitney p=1.89e-08 on a spurious-feature MLP.
- **Honest scope**: Method *fails* on pretrained-ResNet Waterbirds across three variants. We characterize the boundary and recommend bi-level for fine-tuning regimes.
- **Most remarkable result**: lowest seed variance among all methods (test std 0.039) on the Colored MNIST benchmark — fairds-2 is *more stable* than vanilla and Ren2018 in its native regime.

### §1 Introduction (≈1 page)

- **Opening hook**: Spurious correlations make ERM models latch onto majority-group shortcuts; standard remedy is bi-level meta-learning (Ren et al. 2018; FORML), which is implicit-differentiation-heavy.
- **Gap**: Wang et al. 2024 derived a closed-form 1st- and 2nd-order Shapley estimator computable in a single training run, but used it for *post-hoc* data filtering. Whether the same formulas can serve as an *in-training* per-sample reweighter — particularly the 2nd-order gradient-Hessian-gradient cross-term — is open.
- **One-sentence contribution**: see top.
- **Approach**: 5-step pipeline (per-sample grad via vmap → φ via ⟨g_i,g_val⟩ + α⟨g_i, H_val g_val⟩ → EMA → softmax weight → weighted SGD), with **RMS rescaling of the cross-term** as the algorithmic contribution that keeps it stable.
- **Key questions**:
  1. Does the 2nd-order cross-term provide a stand-alone benefit over the 1st-order term?
  2. Where does this method beat / match / lose to bi-level meta-learning?
- **Contributions** (numbered, falsifiable):
  1. Adapt closed-form In-Run Shapley as in-training reweighter; introduce RMS-rescaled cross-term to prevent divergence under strong spurious bias.
  2. Show statistically significant OOD worst-group accuracy improvement on Colored MNIST (+13 pp vs vanilla, +4.7 pp isolation of 2nd-order).
  3. Characterize the regime boundary: from-scratch / small-model spurious works; pretrained-ResNet fine-tuning fails (3 variants tried).
- **Hero figure (Fig 1)**: 3-panel —
  - (a) Algorithm sketch (3 boxes): per-sample grads → φ_1 + RMS-cross → EMA → softmax → weighted SGD.
  - (b) Colored MNIST result bar plot: vanilla / Ren2018 / fairds-1 / fairds-2 worst-group acc with error bars.
  - (c) Waterbirds result bar plot: same methods, fairds bars in red (failure indicator), Ren2018 in green.
- **Front-loading check**: skim reader sees title + abstract + Fig 1 → already knows (i) what we propose, (ii) where it works (Colored MNIST), (iii) where it doesn't (Waterbirds).

### §2 Related Work (≈0.5 page, 3 paragraphs)

- **Para 1 — In-run data valuation**: Wang et al. 2024 In-Run Data Shapley (closed-form, post-hoc filtering); CHG Shapley (Cai 2024, contemporaneous, accuracy-focused). We reuse the formula but re-purpose it for in-training reweighting.
- **Para 2 — Reweighting for fairness/robustness**: Ren et al. 2018 (1-step bi-level), FORML (Yan et al. 2022, fairness extension), FairShap (Arnaiz-Rodriguez et al. 2023, static Shapley preprocessing), ARL (Lahoti et al. 2020, no-demographics adversarial). We explicitly compare against Ren2018 (corrected) and inherit FairShap's "Shapley for fairness" intuition but make it dynamic and closed-form.
- **Para 3 — Spurious-correlation benchmarks**: Colored MNIST (Arjovsky et al. 2019 IRM setup), Waterbirds (Sagawa et al. 2020 GroupDRO).
- **Positioning**: We are *not* a state-of-the-art robustness method on real images; we are a mechanism paper showing when closed-form reweighting suffices vs when bi-level is necessary.

### §3 Method (≈1 page)

- **Notation**: θ_t parameters at step t; D_val anchor (50:50 group-balanced, no sensitive labels needed); per-sample loss L_i.
- **Background**: Wang 2024 Taylor expansion of utility ΔU_i ≈ −η⟨g_i, g_val⟩ + (η²/2)⟨g_i, H_val g_val⟩.
- **Fairds-1**: φ_i^(1) = ⟨g_i, g_val⟩.
- **Fairds-2**: φ_i^(2) = φ_i^(1) − α · ⟨g_i, H_val g_val⟩ where the cross-term is **RMS-rescaled** to first-order's RMS magnitude per minibatch. (One-paragraph derivation of why rescaling is needed: H_val spectrum varies by orders of magnitude across training, so a fixed α cannot survive without rescaling.)
- **Pipeline**: per-sample g via `torch.func.vmap(grad)` → HVP via Pearlmutter trick → φ → EMA(momentum=0.9) → softmax(EMA/τ) → clip → weighted SGD.
- **Implementation note**: BatchNorm running stats frozen for vmap compatibility.
- **Estimated length**: 1 page (algorithm box + 1 paragraph derivation + 1 paragraph hyperparameters).

### §4 Experiments (≈2 pages)

#### §4.1 Colored MNIST (POSITIVE — main result)

- **Setup**: standard Arjovsky 2019. 5000 train (90:10 majority:minority by spurious correlation strength). Anchor 200 (50:50, used inside training). Val_eval 500 (held-out). Test 5000 (OOD, P(c=y)=0.1, group = color-aligned vs flipped). Small CNN. 10 seeds × 20 epochs. Per-method best-tuned lr ∈ {0.005, 0.01, 0.02, 0.05}.
- **Table 1** (main result): vanilla / fairds-1 / fairds-2 / Ren2018(corrected) × val_eval_worst, test_acc, test_worst (mean ± std).
- **Table 2** (paired t-tests): Δ vs vanilla, Δ vs Ren2018, isolation Δ (fairds-2 vs fairds-1).
- **Headline numbers**: fairds-2 test_worst = 0.824 ± 0.043; +13 pp vs vanilla (p=5e-5); +2.7 pp vs Ren2018 (p=0.052); +4.7 pp 2nd-order isolation (p=0.033). fairds-2 has the lowest seed variance.

#### §4.2 Mechanism Verification (E1b spurious-feature MLP)

- **Setup**: 2-layer MLP, controlled spurious feature (s=10 in majority, 0 in minority). 5 seeds × 4 imbalance ratios.
- **Result**: Mann-Whitney U(majority-φ < minority-φ) at ratio 0.9 yields **p = 1.89e-08**; Δφ monotone-decreasing with imbalance.
- **Figure 2**: per-group φ distributions stratified by imbalance ratio.

#### §4.3 Failure regime: Waterbirds (NEGATIVE — honest)

- **Setup**: pretrained ResNet-18, BN frozen for vmap, 4795 train, anchor 200 (50/group), test 5794 (4 groups). 3 seeds × 10–15 epochs.
- **Three algorithmic variants tried**:
  1. default EMA (τ=0.5, ws=1.0): fairds-2 test_worst = 0.145 (catastrophic; below vanilla 0.182).
  2. + warmup_epochs=3: 0.200 (≈ vanilla).
  3. no_ema (per-step φ) + lr=1e-4 + warmup=1: 0.189 (vanilla +7 pp, p=0.17 NS).
- **Reference**: corrected Ren2018 = 0.494 — strictly dominates fairds-2 by Δ=−30 pp (p=0.997).
- **Diagnosis**: in fine-tuning regime, g_i is small/noisy in early epochs → EMA buffer fills with noise → reweighting destabilizes training. No-EMA variant doesn't cure it because the fundamental issue is *low signal-to-noise ratio of g_i during fine-tuning*.
- **Figure 3**: Waterbirds Pareto trade-off (or 4-method × 3-variant matrix).

#### §4.4 Boundary / tabular check (E2 Adult/COMPAS) — Appendix

- 1-paragraph in main body: "On Adult/COMPAS the Pareto front is jointly spanned by vanilla and Ren2018; fairds-2 is dominated; details in Appendix B."

### §5 Discussion: Regime Analysis (≈0.5 page)

- **Where Fairds wins**: from-scratch small-medium models with structural spurious correlations (Colored MNIST, MLP). The 2nd-order cross-term provides isolated benefit + low variance.
- **Where Fairds loses**: pretrained large-model fine-tuning (Waterbirds, Adult/COMPAS where vanilla is already strong).
- **Why**: closed-form Shapley relies on the per-sample gradient being a *signal-rich* indicator of sample utility. Pretrained features make g_i small/noisy → EMA-based reweighting amplifies noise rather than signal.
- **Practical recommendation**: For pretrained fine-tuning, use bi-level meta-learning (Ren et al. 2018). For from-scratch training under spurious correlation, closed-form Shapley is competitive and cheaper per step (no implicit differentiation).
- **Open**: can a hybrid "closed-form for 1st-order, bi-level for cross-term-equivalent correction" work? Out of scope.

### §6 Conclusion (≈0.25 page)

- 2-3 sentence restatement.
- Limitations: scale (small CNN/MLP), reliance on D_val anchor (we don't claim sensitive-label-free robust on real data), one positive benchmark.
- Future: scaling study, theoretical bound on cross-term attenuation, hybrid approach.

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| **Fig 1 (hero)** | Multi-panel: (a) algorithm sketch, (b) CMNIST result bars, (c) Waterbirds result bars | (a) manual TikZ/figurespec; (b)(c) results JSON | HIGH |
| Fig 2 | Per-group φ distribution stratified by imbalance ratio (E1b mechanism) | results/e1b/20260506-153909/sweep_results.json | HIGH |
| Fig 3 | Waterbirds 4-method × 3-variant matrix (test_worst with error bars) | results/e3b{,_warmup3,_no_ema}/sweep_results.json | HIGH |
| **Table 1** | CMNIST main results (4 methods × 3 metrics, mean ± std) | results/e3/20260506-190605/sweep_results.json | HIGH |
| **Table 2** | CMNIST paired t-tests (Δ vs vanilla, vs Ren2018, 2nd-order isolation) | same | HIGH |
| Fig 4 (appendix) | E2 Adult Pareto trade-off (vanilla / fairds-2 grid / ren2018 grid) | results/e2/pareto-20260506-181137/pareto.json | MEDIUM |
| Table 3 (appendix) | E1b ratio × method × phi statistics | results/e1b/...sweep_results.json | MEDIUM |

**Hero Figure 1 — detailed description**:
- 3 panels horizontal arrangement (consistent with workshop 5-page constraint).
- Panel (a) "Method sketch": 4 small boxes left-to-right — (i) per-sample grads g_i + g_val (ii) φ via 1st + RMS-rescaled cross (iii) EMA buffer (iv) softmax weights → weighted SGD.
- Panel (b) "Where it works (Colored MNIST)": grouped bar chart, x = method, y = OOD test_worst. fairds-2 highlighted in green at top (0.824). Vanilla (0.694) and ren2018 (0.796) for comparison. Δ=+13 pp annotated with significance star.
- Panel (c) "Where it fails (Waterbirds)": same layout, fairds-2 in *red* at 0.200 (below vanilla 0.182), ren2018 in green at 0.494. Δ=-30 pp annotated.
- Caption: "Closed-form 2nd-order in-run Shapley reweighting (Fairds-2) improves worst-group accuracy on Colored MNIST by 13 pp but fails on pretrained-ResNet Waterbirds. We characterize the regime boundary."

---

## Citation Plan

(Citations come from `research-wiki/papers/`. All 6 paper entries already validated.)

**§1 Intro**:
- `wang2024_data_shapley_one` (closed-form In-Run Shapley)
- `sagawa2020_groupdro` *(NEW — needs to add to wiki/bib; Waterbirds source)*
- `arjovsky2019_irm` *(NEW — Colored MNIST setup origin)*

**§2 Related Work**:
- `wang2024_data_shapley_one`, `cai2024_chg_shapley_singletrainingrun` (in-run valuation)
- `ren2018_learning_reweight_examples`, `yan2022_forml_fairness_optimized` (bi-level reweighting)
- `arnaizrodriguez2023_fairshap_shapley_value` (static Shapley fairness)
- `lahoti2020_fairness_without_demographics` (no-demographics)
- `arjovsky2019_irm`, `sagawa2020_groupdro` (spurious benchmarks)

**§3 Method**:
- `pearlmutter1994_hvp` *(NEW — HVP trick)*

**§4 Experiments**:
- `arjovsky2019_irm` (Colored MNIST setup)
- `sagawa2020_groupdro` (Waterbirds)
- UCI Adult / ProPublica COMPAS (URL footnote, no formal cite needed)

**Verify list (all `[VERIFY]` until pulled into bib)**:
- arjovsky2019_irm — *Invariant Risk Minimization*, Arjovsky, Bottou, Gulrajani, Lopez-Paz. arXiv:1907.02893. 2019.
- sagawa2020_groupdro — *Distributionally Robust Neural Networks for Group Shifts*, Sagawa, Koh, Hashimoto, Liang. ICLR 2020.
- pearlmutter1994_hvp — *Fast Exact Multiplication by the Hessian*, Pearlmutter. Neural Computation 6(1), 1994.

---

## Reviewer Feedback (placeholder, to be filled by Step 6)

*Will be populated by gpt-5.5 xhigh review of this plan.*

---

## Next Steps
- [ ] `/paper-figure "PAPER_PLAN.md"` — generate Fig 2-4, Tables 1-3 from results JSON
- [ ] `/figure-spec` — generate hero Fig 1 (algorithm sketch + bar charts) deterministically
- [ ] `/paper-write "PAPER_PLAN.md"` — section-by-section LaTeX
- [ ] `/paper-compile "paper/"` — build PDF
- [ ] `/auto-paper-improvement-loop "paper/"` — 2 rounds polishing

**Estimated total time**: 60-90 min from here to compiled PDF.
