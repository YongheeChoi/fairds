# E2 — Adult / COMPAS sweep summary

## Per-method × dataset × val-mode (mean ± std over seeds)

| dataset | method | val_mode | val_acc | worst_acc | dp_diff | eo_diff | walltime |
|---|---|---|---|---|---|---|---|
| adult | fairds-1 | balanced | 0.835±0.001 | 0.796±0.001 | 0.185±0.017 | 0.127±0.043 | 7.54s |
| adult | fairds-1 | random | 0.836±0.001 | 0.796±0.001 | 0.190±0.018 | 0.109±0.032 | 7.55s |
| adult | fairds-2 | balanced | 0.835±0.001 | 0.796±0.001 | 0.189±0.015 | 0.127±0.042 | 9.97s |
| adult | fairds-2 | random | 0.836±0.001 | 0.797±0.001 | 0.194±0.014 | 0.111±0.028 | 9.78s |
| adult | vanilla | balanced | 0.837±0.001 | 0.799±0.001 | 0.196±0.004 | 0.117±0.015 | 3.16s |
| adult | vanilla | random | 0.837±0.001 | 0.799±0.001 | 0.196±0.004 | 0.117±0.015 | 2.95s |
| compas | fairds-1 | balanced | 0.667±0.008 | 0.656±0.014 | 0.238±0.018 | 0.240±0.027 | 0.76s |
| compas | fairds-1 | random | 0.659±0.013 | 0.648±0.016 | 0.244±0.013 | 0.233±0.022 | 0.76s |
| compas | fairds-2 | balanced | 0.671±0.004 | 0.663±0.007 | 0.239±0.021 | 0.244±0.023 | 0.94s |
| compas | fairds-2 | random | 0.665±0.008 | 0.654±0.015 | 0.246±0.020 | 0.237±0.029 | 0.94s |
| compas | vanilla | balanced | 0.667±0.013 | 0.656±0.019 | 0.240±0.025 | 0.226±0.025 | 0.31s |
| compas | vanilla | random | 0.667±0.013 | 0.656±0.019 | 0.240±0.025 | 0.226±0.025 | 0.33s |

## C2 — Δ vs Vanilla (paired t-test, balanced val_mode)

| dataset | method | Δval_acc | p_acc | Δdp_diff | p_dp | Δeo_diff | p_eo |
|---|---|---|---|---|---|---|---|
| adult | fairds-1 | -0.0026 | 0.0055 | -0.0109 | 0.2487 | 0.0098 | 0.6259 |
| adult | fairds-2 | -0.0025 | 0.0008 | -0.0068 | 0.3852 | 0.0098 | 0.6136 |
| compas | fairds-1 | -0.0008 | 0.9429 | -0.0020 | 0.8896 | 0.0143 | 0.4602 |
| compas | fairds-2 | 0.0034 | 0.6842 | -0.0014 | 0.9335 | 0.0178 | 0.3663 |

## C4 — Sensitive-attribute-free anchor (balanced vs random val_mode)

Δ(metric, random − balanced) for each method/dataset. 
If Fairds is robust to lack of sensitive labels, fairness metrics should be similar across val modes.

| dataset | method | Δval_acc (rand-bal) | Δdp_diff | Δeo_diff |
|---|---|---|---|---|
| adult | fairds-1 | 0.0010 | 0.0046 | -0.0179 |
| adult | fairds-2 | 0.0011 | 0.0046 | -0.0156 |
| adult | vanilla | 0.0000 | 0.0000 | 0.0000 |
| compas | fairds-1 | -0.0074 | 0.0053 | -0.0071 |
| compas | fairds-2 | -0.0057 | 0.0072 | -0.0067 |
| compas | vanilla | 0.0000 | 0.0000 | 0.0000 |

## C4 detail — fairness recovery rate
Recovery rate := (vanilla_metric − fairds_metric) / vanilla_metric, evaluated separately on balanced vs random anchors.

Larger positive value = stronger improvement vs vanilla. If fairds-* recovery_random / recovery_balanced ≥ 0.9, C4 is supported.

| dataset | method | metric | rec(balanced) | rec(random) | rec_random / rec_balanced |
|---|---|---|---|---|---|
| adult | fairds-1 | dp_diff | 5.6% | 3.2% | 57.5% |
| adult | fairds-2 | dp_diff | 3.4% | 1.1% | 32.2% |
| adult | fairds-1 | eo_diff | -8.4% | 7.0% | -83.2% |
| adult | fairds-2 | eo_diff | -8.4% | 5.0% | -59.7% |
| compas | fairds-1 | dp_diff | 0.8% | -1.4% | -165.8% |
| compas | fairds-2 | dp_diff | 0.6% | -2.4% | -414.3% |
| compas | fairds-1 | eo_diff | -6.3% | -3.2% | 50.5% |
| compas | fairds-2 | eo_diff | -7.9% | -5.0% | 62.7% |
