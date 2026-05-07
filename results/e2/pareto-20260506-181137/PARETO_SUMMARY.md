# E2 Adult — Pareto trade-off (fairds-2 grid vs ren2018 grid vs vanilla)

Mean over 3 seeds. **Bold** = Pareto-optimal in (acc, dp+eo) space (lower dp/eo better, higher acc better).

| config | method | val_acc | worst | dp_diff | eo_diff | dp+eo | w_std |
|---|---|---|---|---|---|---|---|
| **vanilla** | vanilla | 0.837±0.001 | 0.798 | 0.199±0.002 | 0.125±0.011 | 0.324 | — |
| fairds-2/τ1.0/ws1.0 | fairds-2 | 0.835±0.002 | 0.797 | 0.196±0.009 | 0.149±0.014 | 0.345 | 0.108 |
| fairds-2/τ0.5/ws1.0 | fairds-2 | 0.834±0.001 | 0.796 | 0.196±0.015 | 0.157±0.023 | 0.353 | 0.190 |
| fairds-2/τ1.0/ws2.0 | fairds-2 | 0.834±0.002 | 0.795 | 0.196±0.015 | 0.156±0.026 | 0.352 | 0.184 |
| fairds-2/τ0.3/ws1.0 | fairds-2 | 0.833±0.001 | 0.794 | 0.196±0.016 | 0.158±0.030 | 0.354 | 0.293 |
| fairds-2/τ0.5/ws2.0 | fairds-2 | 0.833±0.001 | 0.793 | 0.196±0.018 | 0.155±0.029 | 0.351 | 0.331 |
| fairds-2/τ1.0/ws4.0 | fairds-2 | 0.832±0.001 | 0.793 | 0.196±0.018 | 0.151±0.032 | 0.347 | 0.329 |
| fairds-2/τ0.3/ws2.0 | fairds-2 | 0.831±0.002 | 0.791 | 0.196±0.023 | 0.152±0.039 | 0.349 | 0.521 |
| fairds-2/τ0.5/ws4.0 | fairds-2 | 0.829±0.002 | 0.789 | 0.195±0.025 | 0.148±0.048 | 0.342 | 0.587 |
| fairds-2/τ0.1/ws1.0 | fairds-2 | 0.829±0.003 | 0.789 | 0.195±0.029 | 0.150±0.059 | 0.345 | 0.851 |
| **ren2018/lr0.05** | ren2018 | 0.827±0.004 | 0.788 | 0.188±0.029 | 0.134±0.069 | 0.323 | — |
| fairds-2/τ0.3/ws4.0 | fairds-2 | 0.824±0.006 | 0.784 | 0.192±0.034 | 0.158±0.081 | 0.350 | 0.881 |
| ren2018/lr0.02 | ren2018 | 0.823±0.007 | 0.784 | 0.186±0.029 | 0.140±0.069 | 0.326 | — |
| ren2018/lr0.01 | ren2018 | 0.820±0.007 | 0.780 | 0.191±0.028 | 0.162±0.058 | 0.353 | — |
| fairds-2/τ0.1/ws2.0 | fairds-2 | 0.819±0.007 | 0.778 | 0.191±0.037 | 0.160±0.086 | 0.352 | 1.404 |
| ren2018/lr0.005 | ren2018 | 0.815±0.006 | 0.775 | 0.191±0.029 | 0.185±0.052 | 0.377 | — |
| fairds-2/τ0.1/ws4.0 | fairds-2 | 0.814±0.009 | 0.773 | 0.190±0.035 | 0.161±0.078 | 0.351 | 1.840 |

## Pareto-optimal configurations (high acc AND low dp+eo)

| config | method | acc | dp_diff | eo_diff | dp+eo |
|---|---|---|---|---|---|
| vanilla | vanilla | 0.837 | 0.199 | 0.125 | 0.324 |
| ren2018/lr0.05 | ren2018 | 0.827 | 0.188 | 0.134 | 0.323 |

## Best per-method (lowest dp+eo subject to acc ≥ vanilla − 1pp)

vanilla acc = 0.837, threshold = 0.827

| method | best config | acc | dp_diff | eo_diff | dp+eo | Δ vs vanilla(dp+eo) |
|---|---|---|---|---|---|---|
| fairds-2 | fairds-2/τ0.5/ws4.0 | 0.829 | 0.195 | 0.148 | 0.342 | +0.018 |
| ren2018 | ren2018/lr0.05 | 0.827 | 0.188 | 0.134 | 0.323 | -0.001 |

## Diagnostic: weight magnitude (fairds-2 only)

Higher w_std means stronger reweighting. Codex Round 1 noted Δw was tiny — does τ↓ or weight_scale↑ fix it?

| config | w_std | acc | dp+eo |
|---|---|---|---|
| fairds-2/τ0.1/ws4.0 | 1.840 | 0.814 | 0.351 |
| fairds-2/τ0.1/ws2.0 | 1.404 | 0.819 | 0.352 |
| fairds-2/τ0.3/ws4.0 | 0.881 | 0.824 | 0.350 |
| fairds-2/τ0.1/ws1.0 | 0.851 | 0.829 | 0.345 |
| fairds-2/τ0.5/ws4.0 | 0.587 | 0.829 | 0.342 |
| fairds-2/τ0.3/ws2.0 | 0.521 | 0.831 | 0.349 |
| fairds-2/τ0.5/ws2.0 | 0.331 | 0.833 | 0.351 |
| fairds-2/τ1.0/ws4.0 | 0.329 | 0.832 | 0.347 |
| fairds-2/τ0.3/ws1.0 | 0.293 | 0.833 | 0.354 |
| fairds-2/τ0.5/ws1.0 | 0.190 | 0.834 | 0.353 |
| fairds-2/τ1.0/ws2.0 | 0.184 | 0.834 | 0.352 |
| fairds-2/τ1.0/ws1.0 | 0.108 | 0.835 | 0.345 |