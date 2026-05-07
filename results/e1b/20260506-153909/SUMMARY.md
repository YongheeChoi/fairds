# E1b — Spurious-feature 2-group MLP sweep summary

## Worst-group accuracy (primary metric)

| method | ratio | val_acc | worst | val_acc_g0 (maj) | val_acc_g1 (min) | dp_diff | eo_diff |
|---|---|---|---|---|---|---|---|
| fairds-1 | 0.50 | 0.811 | 0.621 | 1.000 | 0.621 | 0.043 | 0.336 |
| fairds-1 | 0.70 | 0.810 | 0.620 | 1.000 | 0.620 | 0.033 | 0.349 |
| fairds-1 | 0.90 | 0.779 | 0.559 | 1.000 | 0.559 | 0.079 | 0.368 |
| fairds-1 | 0.99 | 0.759 | 0.519 | 1.000 | 0.519 | 0.119 | 0.363 |
| fairds-2 | 0.50 | 0.815 | 0.629 | 1.000 | 0.629 | 0.048 | 0.325 |
| fairds-2 | 0.70 | 0.813 | 0.627 | 1.000 | 0.627 | 0.048 | 0.336 |
| fairds-2 | 0.90 | 0.785 | 0.571 | 1.000 | 0.571 | 0.096 | 0.333 |
| fairds-2 | 0.99 | 0.766 | 0.532 | 1.000 | 0.532 | 0.105 | 0.365 |
| vanilla | 0.50 | 0.810 | 0.620 | 1.000 | 0.620 | 0.063 | 0.320 |
| vanilla | 0.70 | 0.811 | 0.621 | 1.000 | 0.621 | 0.069 | 0.328 |
| vanilla | 0.90 | 0.779 | 0.559 | 1.000 | 0.559 | 0.108 | 0.333 |
| vanilla | 0.99 | 0.764 | 0.528 | 1.000 | 0.528 | 0.117 | 0.373 |

## Δworst (method − vanilla), paired across seeds

| method | ratio | mean Δworst | t-stat | p (one-sided > 0) |
|---|---|---|---|---|
| fairds-1 | 0.50 | 0.0013 | 0.123 | 0.4540 |
| fairds-1 | 0.70 | -0.0013 | -0.232 | 0.5862 |
| fairds-1 | 0.90 | -0.0000 | -0.000 | 0.5000 |
| fairds-1 | 0.99 | -0.0093 | -1.360 | 0.8773 |
| fairds-2 | 0.50 | 0.0093 | 0.704 | 0.2603 |
| fairds-2 | 0.70 | 0.0053 | 1.089 | 0.1688 |
| fairds-2 | 0.90 | 0.0120 | 2.092 | 0.0523 |
| fairds-2 | 0.99 | 0.0040 | 1.177 | 0.1523 |

## Isolation of 2nd-order: Δworst (fairds-2 − fairds-1)

| ratio | fairds-1 worst | fairds-2 worst | Δ |
|---|---|---|---|
| 0.50 | 0.621 | 0.629 | 0.0080 |
| 0.70 | 0.620 | 0.627 | 0.0067 |
| 0.90 | 0.559 | 0.571 | 0.0120 |
| 0.99 | 0.519 | 0.532 | 0.0133 |

## Per-group Shapley value (mean across seeds)
Δφ = mean(φ | majority) − mean(φ | minority); negative = majority gets DOWN-weighted (C3 expected).

| method | ratio | mean_phi_maj | mean_phi_min | Δφ | mean_w_maj | mean_w_min | Δw |
|---|---|---|---|---|---|---|---|
| fairds-1 | 0.50 | 0.00005 | -0.00216 | 0.00222 | 1.0557 | 0.9443 | 0.1114 |
| fairds-1 | 0.70 | -0.00008 | 0.00552 | -0.00560 | 1.0369 | 0.9140 | 0.1229 |
| fairds-1 | 0.90 | -0.00013 | 0.01494 | -0.01507 | 1.0091 | 0.9185 | 0.0906 |
| fairds-1 | 0.99 | -0.00014 | 0.03709 | -0.03723 | 1.0007 | 0.9319 | 0.0688 |
| fairds-2 | 0.50 | 0.00019 | -0.00180 | 0.00199 | 1.0246 | 0.9754 | 0.0492 |
| fairds-2 | 0.70 | -0.00000 | 0.00278 | -0.00279 | 1.0122 | 0.9716 | 0.0405 |
| fairds-2 | 0.90 | -0.00006 | 0.01134 | -0.01140 | 1.0021 | 0.9811 | 0.0210 |
| fairds-2 | 0.99 | -0.00009 | 0.02508 | -0.02517 | 1.0001 | 0.9933 | 0.0067 |

## C3 hypothesis test (Mann-Whitney U, fairds-2, ratio=0.9)

- U=317454, p (one-sided maj < min)=1.89e-08, n_maj=1000, n_min=750
- → C3(a) SUPPORTED at α=0.05

## Wall-clock per-run (mean over seeds, ratio=0.9)

- fairds-1: 2.18s
- fairds-2: 2.76s
- vanilla: 0.41s