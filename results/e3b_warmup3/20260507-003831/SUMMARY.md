# E3b — Waterbirds sweep summary

Real-image spurious-correlation benchmark (Sagawa et al. 2020).
Held-out worst-group acc on val_eval (anchor-disjoint) and test (true OOD setting w/ 4 group breakdown).

## Per-method aggregates (mean ± std over seeds)

| method | val_eval_worst | test_acc | test_worst |
|---|---|---|---|
| vanilla | 0.197±0.020 | 0.657±0.017 | 0.182±0.033 |
| fairds-1 | 0.191±0.073 | 0.583±0.067 | 0.180±0.081 |
| fairds-2 | 0.210±0.080 | 0.581±0.059 | 0.200±0.086 |
| ren2018 | 0.480±0.052 | 0.685±0.039 | 0.494±0.058 |

## Paired tests vs vanilla (one-sided > 0)

| method | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-1 | -0.0057 | 0.5373 | -0.0735 | 0.9132 | -0.0022 | 0.5126 |
| fairds-2 | +0.0134 | 0.4179 | -0.0755 | 0.9354 | +0.0179 | 0.4084 |
| ren2018 | +0.2836 | 0.01544 | +0.0278 | 0.2202 | +0.3115 | 0.01953 |

## Isolation tests

| comparison | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-2 vs fairds-1 | +0.0191 | 0.08141 | -0.0020 | 0.5907 | +0.0202 | 0.06255 |
| fairds-2 vs ren2018 | -0.2702 | 0.9746 | -0.1033 | 0.9073 | -0.2935 | 0.9808 |

## Per-seed test_worst details

| seed | vanilla | fairds-1 | fairds-2 | ren2018 |
|---|---|---|---|---|
| 0 | 0.213 | 0.263 | 0.277 | 0.455 |
| 1 | 0.137 | 0.207 | 0.243 | 0.575 |
| 2 | 0.196 | 0.070 | 0.080 | 0.451 |
