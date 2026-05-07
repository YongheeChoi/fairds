# E3 — Colored MNIST sweep summary (post Round 3 fix)

Standard spurious-correlation benchmark per Codex Round 3 review.

Setup: 5000 train samples, ratio=0.9, 10 seeds, 20 epochs, τ=0.1, ws=4.0.

## Per-method aggregates (mean ± std over 10 seeds)

| method | val_acc | val_worst | test_acc |
|---|---|---|---|
| vanilla | 0.690±0.014 | 0.486±0.024 | 0.098±0.004 |
| fairds-1 | 0.708±0.025 | 0.534±0.062 | 0.194±0.100 |
| fairds-2 | 0.709±0.028 | 0.542±0.067 | 0.223±0.121 |
| ren2018 | 0.700±0.021 | 0.502±0.037 | 0.120±0.031 |

## Paired tests (each method vs vanilla, same seeds)

| method | Δval_worst | t (paired) | p (one-sided > 0) | Δtest_acc | t | p |
|---|---|---|---|---|---|---|
| fairds-1 | +0.0480 | 2.83 | 0.009802 | +0.0960 | 2.90 | 0.008797 |
| fairds-2 | +0.0568 | 3.20 | 0.005426 | +0.1253 | 3.12 | 0.006146 |
| ren2018 | +0.0164 | 2.11 | 0.03221 | +0.0221 | 2.02 | 0.03679 |

## Isolation tests (fairds-2 vs fairds-1, fairds-2 vs ren2018)

| comparison | Δval_worst | t | p (one-sided > 0) | Δtest_acc | t | p |
|---|---|---|---|---|---|---|
| fairds-2 vs fairds-1 | +0.0088 | 1.82 | 0.0511 | +0.0293 | 3.45 | 0.003654 |
| fairds-2 vs ren2018 | +0.0404 | 2.42 | 0.01924 | +0.1032 | 2.65 | 0.01326 |

## Per-seed details (val_worst)

| seed | vanilla | fairds-1 | fairds-2 | ren2018 |
|---|---|---|---|---|
| 0 | 0.472 | 0.472 | 0.472 | 0.472 |
| 1 | 0.516 | 0.588 | 0.612 | 0.596 |
| 2 | 0.452 | 0.452 | 0.452 | 0.452 |
| 3 | 0.492 | 0.568 | 0.556 | 0.516 |
| 4 | 0.504 | 0.504 | 0.504 | 0.504 |
| 5 | 0.440 | 0.456 | 0.452 | 0.468 |
| 6 | 0.512 | 0.608 | 0.624 | 0.520 |
| 7 | 0.496 | 0.496 | 0.532 | 0.496 |
| 8 | 0.476 | 0.632 | 0.636 | 0.492 |
| 9 | 0.496 | 0.560 | 0.584 | 0.504 |
