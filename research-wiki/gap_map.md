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

The 2nd-order Shapley gradient-Hessian-gradient cross-term may automatically attenuate representation bias via majority-group information overlap — but this potential is not analyzed theoretically or empirically in prior work, including the original In-Run Data Shapley paper which treats it only as a computational artifact.

  - Related papers: paper:wang2024_data_shapley_one
