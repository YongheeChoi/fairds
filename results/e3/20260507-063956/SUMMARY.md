# E3 — Colored MNIST sweep summary (post Round 3 fix)

Standard spurious-correlation benchmark per Codex Round 3 review.

Setup: 5000 train samples, ratio=0.9, 10 seeds, 20 epochs, τ=0.1, ws=4.0.

## Per-method aggregates (mean ± std over 10 seeds, held-out)

| method | val_eval_worst | test_acc | test_worst |
|---|---|---|---|
| vanilla | 0.802±0.068 | 0.716±0.066 | 0.694±0.064 |
| fairds-1 | 0.842±0.057 | 0.791±0.069 | 0.777±0.071 |
| fairds-2 | 0.878±0.034 | 0.836±0.039 | 0.824±0.043 |
| ren2018 | 0.878±0.020 | 0.813±0.048 | 0.796±0.054 |

## Paired tests (each method vs vanilla, same seeds)

| method | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-1 | +0.0392 | 0.08526 | +0.0757 | 0.0004418 | +0.0837 | 0.0001498 |
| fairds-2 | +0.0756 | 0.009393 | +0.1203 | 0.0001025 | +0.1302 | 5.377e-05 |
| ren2018 | +0.0760 | 0.005264 | +0.0975 | 4.485e-06 | +0.1029 | 1.593e-06 |

## Isolation tests (fairds-2 vs fairds-1, fairds-2 vs ren2018)

| comparison | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-2 vs fairds-1 | +0.0364 | 0.01195 | +0.0446 | 0.0311 | +0.0465 | 0.03339 |
| fairds-2 vs ren2018 | -0.0004 | 0.5179 | +0.0228 | 0.05666 | +0.0272 | 0.05228 |

## Per-seed details (val_worst)

| seed | vanilla | fairds-1 | fairds-2 | ren2018 |
|---|---|---|---|---|
| 0 | 0.808 | 0.868 | 0.844 | 0.856 |
| 1 | 0.876 | 0.868 | 0.852 | 0.888 |
| 2 | 0.800 | 0.780 | 0.872 | 0.856 |
| 3 | 0.856 | 0.864 | 0.896 | 0.912 |
| 4 | 0.720 | 0.708 | 0.808 | 0.856 |
| 5 | 0.820 | 0.800 | 0.880 | 0.868 |
| 6 | 0.844 | 0.888 | 0.908 | 0.880 |
| 7 | 0.640 | 0.900 | 0.936 | 0.904 |
| 8 | 0.812 | 0.864 | 0.892 | 0.896 |
| 9 | 0.848 | 0.876 | 0.892 | 0.868 |
