# E3b — Waterbirds sweep summary

Real-image spurious-correlation benchmark (Sagawa et al. 2020).
Held-out worst-group acc on val_eval (anchor-disjoint) and test (true OOD setting w/ 4 group breakdown).

## Per-method aggregates (mean ± std over seeds)

| method | val_eval_worst | test_acc | test_worst |
|---|---|---|---|
| vanilla | 0.096±0.035 | 0.702±0.017 | 0.115±0.064 |
| fairds-1 | 0.198±0.057 | 0.532±0.030 | 0.184±0.050 |
| fairds-2 | 0.199±0.053 | 0.535±0.028 | 0.189±0.047 |
| ren2018 | 0.506±0.082 | 0.629±0.057 | 0.499±0.081 |

## Paired tests vs vanilla (one-sided > 0)

| method | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-1 | +0.1015 | 0.09156 | -0.1709 | 0.994 | +0.0696 | 0.1735 |
| fairds-2 | +0.1023 | 0.08811 | -0.1677 | 0.9938 | +0.0745 | 0.1681 |
| ren2018 | +0.4100 | 0.01499 | -0.0734 | 0.8939 | +0.3847 | 0.02258 |

## Isolation tests

| comparison | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |
|---|---|---|---|---|---|---|
| fairds-2 vs fairds-1 | +0.0008 | 0.4423 | +0.0032 | 0.2158 | +0.0049 | 0.2059 |
| fairds-2 vs ren2018 | -0.3077 | 0.9973 | -0.0943 | 0.9751 | -0.3103 | 0.9965 |

## Per-seed test_worst details

| seed | vanilla | fairds-1 | fairds-2 | ren2018 |
|---|---|---|---|---|
| 0 | 0.202 | 0.203 | 0.198 | 0.487 |
| 1 | 0.051 | 0.234 | 0.242 | 0.604 |
| 2 | 0.090 | 0.116 | 0.127 | 0.407 |
