# E3b — Waterbirds sweep summary

Real-image spurious-correlation benchmark (Sagawa et al. 2020).
Held-out worst-group acc on val_eval (anchor-disjoint) and test (true OOD setting w/ 4 group breakdown).

## Per-method aggregates (mean ± std over seeds)

| method | val_eval_worst | test_acc | test_worst |
|---|---|---|---|
| vanilla | 0.197±0.020 | 0.657±0.017 | 0.182±0.033 |
| fairds-1 | 0.135±0.087 | 0.510±0.062 | 0.134±0.095 |
| fairds-2 | 0.149±0.072 | 0.530±0.051 | 0.145±0.081 |
| ren2018 | 0.480±0.052 | 0.685±0.039 | 0.494±0.058 |

## Paired tests vs vanilla (one-sided > 0)

| method | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-1 | -0.0614 | 0.7487 | -0.1471 | 0.9604 | -0.0483 | 0.6768 |
| fairds-2 | -0.0477 | 0.7347 | -0.1271 | 0.9715 | -0.0372 | 0.6591 |
| ren2018 | +0.2836 | 0.01544 | +0.0278 | 0.2202 | +0.3115 | 0.01953 |

## Isolation tests

| comparison | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-2 vs fairds-1 | +0.0136 | 0.3312 | +0.0200 | 0.1933 | +0.0111 | 0.3419 |
| fairds-2 vs ren2018 | -0.3314 | 0.9983 | -0.1549 | 0.9948 | -0.3487 | 0.9982 |

## Per-seed test_worst details

| seed | vanilla | fairds-1 | fairds-2 | ren2018 |
|---|---|---|---|---|
| 0 | 0.213 | 0.060 | 0.118 | 0.455 |
| 1 | 0.137 | 0.269 | 0.255 | 0.575 |
| 2 | 0.196 | 0.073 | 0.062 | 0.451 |
