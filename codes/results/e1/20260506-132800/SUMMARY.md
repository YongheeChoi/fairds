# E1 Sweep Summary (means over seeds)

## Validation accuracy / DP-diff / EO-diff
| method | ratio | val_acc | val_acc_g0 (maj) | val_acc_g1 (min) | dp_diff | eo_diff | walltime/run |
|---|---|---|---|---|---|---|---|
| fairds-1 | 0.50 | 0.977 | 1.000 | 0.955 | 0.011 | 0.048 | 4.49s |
| fairds-1 | 0.70 | 0.975 | 0.999 | 0.952 | 0.017 | 0.046 | 4.98s |
| fairds-1 | 0.90 | 0.975 | 1.000 | 0.950 | 0.020 | 0.050 | 6.16s |
| fairds-1 | 0.99 | 0.974 | 0.999 | 0.949 | 0.020 | 0.052 | 6.14s |
| fairds-2 | 0.50 | 0.978 | 1.000 | 0.956 | 0.012 | 0.046 | 6.45s |
| fairds-2 | 0.70 | 0.975 | 0.999 | 0.951 | 0.018 | 0.046 | 6.47s |
| fairds-2 | 0.90 | 0.975 | 1.000 | 0.950 | 0.020 | 0.050 | 6.52s |
| fairds-2 | 0.99 | 0.974 | 0.999 | 0.949 | 0.020 | 0.052 | 6.53s |
| vanilla | 0.50 | 0.976 | 1.000 | 0.953 | 0.011 | 0.050 | 0.42s |
| vanilla | 0.70 | 0.974 | 0.999 | 0.949 | 0.016 | 0.050 | 0.26s |
| vanilla | 0.90 | 0.975 | 1.000 | 0.950 | 0.020 | 0.050 | 0.23s |
| vanilla | 0.99 | 0.974 | 0.999 | 0.949 | 0.022 | 0.052 | 0.21s |

## Per-group Shapley value (mean across seeds)
Δφ = mean(φ | majority) − mean(φ | minority).
If C3 holds (representation-bias attenuation), Δφ should decrease (preferably go negative) as imbalance grows for fairds-2.

| method | ratio | mean_phi_majority | mean_phi_minority | Δφ (maj − min) | mean_w_majority | mean_w_minority | Δw |
|---|---|---|---|---|---|---|---|
| fairds-1 | 0.50 | 0.00018 | -0.00008 | 0.00026 | 1.0016 | 0.9984 | 0.0031 |
| fairds-1 | 0.70 | 0.00017 | -0.00049 | 0.00066 | 1.0012 | 0.9971 | 0.0042 |
| fairds-1 | 0.90 | 0.00013 | -0.00102 | 0.00116 | 1.0016 | 0.9859 | 0.0157 |
| fairds-1 | 0.99 | 0.00010 | -0.00137 | 0.00147 | 1.0002 | 0.9804 | 0.0198 |
| fairds-2 | 0.50 | 0.00019 | -0.00007 | 0.00026 | 1.0006 | 0.9994 | 0.0011 |
| fairds-2 | 0.70 | 0.00017 | -0.00046 | 0.00063 | 1.0002 | 0.9995 | 0.0007 |
| fairds-2 | 0.90 | 0.00013 | -0.00099 | 0.00112 | 1.0004 | 0.9965 | 0.0039 |
| fairds-2 | 0.99 | 0.00010 | -0.00135 | 0.00145 | 1.0001 | 0.9929 | 0.0072 |

## C3 hypothesis tests
- (a) For majority_ratio=0.90, test fairds-2 majority-vs-minority phi via Mann–Whitney U on the pooled per-sample values across all 5 seeds.
- (b) Monotone decrease of Δφ with ratio for fairds-2.

- (a) ratio=0.90, Mann–Whitney U(maj < min) for fairds-2: U=562727, p=1, n_maj=1000, n_min=1000
  → **C3(a) NOT SUPPORTED** at α=0.05
- (b) Δφ across ratios for fairds-2: 0.50→+0.00026, 0.70→+0.00063, 0.90→+0.00112, 0.99→+0.00145
  → **C3(b) NOT SUPPORTED**

## fairds-1 vs fairds-2: isolation of the 2nd-order contribution
Compare Δφ for fairds-1 vs fairds-2 across ratios:
| ratio | Δφ (fairds-1) | Δφ (fairds-2) | Δφ₂ − Δφ₁ |
|---|---|---|---|
| 0.50 | +0.00026 | +0.00026 | +0.00000 |
| 0.70 | +0.00066 | +0.00063 | -0.00003 |
| 0.90 | +0.00116 | +0.00112 | -0.00003 |
| 0.99 | +0.00147 | +0.00145 | -0.00002 |

## Wall-clock (rough overhead vs vanilla)
- ratio=0.50: vanilla=0.42s | fairds-1=4.49s (10.78×)| fairds-2=6.45s (15.50×)
- ratio=0.70: vanilla=0.26s | fairds-1=4.98s (19.29×)| fairds-2=6.47s (25.07×)
- ratio=0.90: vanilla=0.23s | fairds-1=6.16s (26.66×)| fairds-2=6.52s (28.22×)
- ratio=0.99: vanilla=0.21s | fairds-1=6.14s (28.57×)| fairds-2=6.53s (30.41×)
