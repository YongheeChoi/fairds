# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
# Gap Map

_Field gaps with stable IDs._

## G1

In-Run Data Shapley has only been applied as a post-hoc data filter; never as an in-run reweighting controller for fairness.

  - Related papers: paper:wang2024_data_shapley_one

## G2

Meta-learning reweighters (FORML, Ren2018) learn weights via gradient descent, providing no theoretical guarantee of meta-fairness — equally contributing samples may receive different weights.

  - Related papers: paper:yan2022_forml_fairness_optimized, paper:ren2018_learning_reweight_examples

## G3

FORML and similar meta-learning reweighters require sensitive-attribute labels on the exemplar/validation set to compute group-fairness objectives — limiting practical use under GDPR-style constraints and labeling costs.

  - Related papers: paper:yan2022_forml_fairness_optimized

## G4

Empirical validation of meta-learning reweighting is restricted to small classification benchmarks (CIFAR / CelebA); real-time bias mitigation at foundation-model scale is unexplored.

  - Related papers: paper:yan2022_forml_fairness_optimized, paper:ren2018_learning_reweight_examples

## G5

The 2nd-order Shapley gradient-Hessian-gradient cross-term may automatically at
## Key Papers (6 total)
- [paper:arnaizrodriguez2023_fairshap_shapley_value] FairShap: Shapley Value Based Fair Data Valuation: 
- [paper:cai2024_chg_shapley_singletrainingrun] CHG Shapley: Single-Training-Run Shapley Value for Sample Efficiency: 
- [paper:lahoti2020_fairness_without_demographics] Fairness without Demographics through Adversarially Reweighted Learning: 
- [paper:ren2018_learning_reweight_examples] Learning to Reweight Examples for Robust Deep Learning: 
- [paper:wang2024_data_shapley_one] Data Shapley in One Training Run: 
- [paper:yan2022_forml_fairness_optimized] FORML: Fairness Optimized Reweighting via Meta-Learning: 
## Recent Relationships (29 total)
  idea:dynamic-in-run-shapley-fairness --addresses_gap--> gap:G4
  idea:dynamic-in-run-shapley-fairness --addresses_gap--> gap:G5
  exp:E1 --tested_by--> claim:C3
  exp:E2 --tested_by--> claim:C2
  exp:E2 --tested_by--> claim:C4
  exp:E3 --tested_by--> claim:C2
  exp:E4 --tested_by--> claim:C1
  exp:E5 --tested_by--> claim:C4
  exp:E5 --tested_by--> claim:C6
  exp:E6 --tested_by--> claim:C5
  exp:E7 --tested_by--> claim:C2
  exp:E1 --invalidates--> claim:C3
  exp:E1 --invalidates--> claim:C1
  exp:E1b --supports--> claim:C3
  exp:E2 --invalidates--> claim:C2
  exp:E2 --invalidates--> claim:C4
  exp:E3 --supports--> claim:C3
  exp:E3 --supports--> claim:C2
  exp:E3b --invalidates--> claim:C2
  exp:E4 --invalidates--> claim:C1
