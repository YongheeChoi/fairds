"""Analyze E7 CelebA fairness results: worst-group, disparity, DP/EO, seed-paired.

CelebA = demographic + STRUCTURED (image) bias. Hypothesis: unlike diffuse tabular
Adult/COMPAS, the closed-form Shapley reweighter works here. We check worst-group
accuracy and the gender DP/EO gaps, with seed-paired tests vs the headline method.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOTS = ["e7_main", "e7_oracle", "e7_resid"]


def load(root):
    base = Path(__file__).resolve().parents[2]
    js = sorted(glob.glob(str(base / f"results/{root}/*/sweep_results.json")))
    return json.load(open(js[-1]))["runs"] if js else []


def main():
    runs = []
    for r in ROOTS:
        runs += load(r)

    keys = ["test_worst", "test_disparity", "test_dp_diff", "test_eo_diff", "test_acc"]
    by = defaultdict(lambda: defaultdict(dict))  # metric -> method -> {seed: val}
    for r in runs:
        m = r["method"] + ("-resid" if r.get("arm") == "residual_real" else "")
        for k in keys:
            if r.get(k) is not None:
                by[k][m][r["seed"]] = r[k]

    order = ["vanilla", "fairds-1", "fairds-2", "fairds-2-resid", "ren2018", "jtt", "groupdro"]
    methods = [m for m in order if m in by["test_worst"]] + \
              [m for m in by["test_worst"] if m not in order]

    print("=" * 84)
    print("E7 CelebA fairness (Blond_Hair target, Male group) — 10 seeds")
    print("worst↑ fairer | disparity↓ fairer | DP↓ fairer | EO↓ fairer")
    print("=" * 84)
    print(f"{'method':16s} {'worst':>12s} {'disparity':>12s} {'DP':>10s} {'EO':>10s} {'acc':>8s}")
    for m in methods:
        def stat(k):
            v = np.array([by[k][m][s] for s in sorted(by[k][m])]) if m in by[k] else np.array([np.nan])
            return v.mean(), v.std()
        w = stat("test_worst"); d = stat("test_disparity")
        dp = stat("test_dp_diff"); eo = stat("test_eo_diff"); a = stat("test_acc")
        print(f"{m:16s} {w[0]:>6.3f}±{w[1]:<5.3f} {d[0]:>6.3f}±{d[1]:<5.3f} "
              f"{dp[0]:>5.3f} {eo[0]:>5.3f} {a[0]:>7.3f}")

    base = "fairds-2-resid" if "fairds-2-resid" in by["test_worst"] else "fairds-2"
    print("\n" + "=" * 84)
    print(f"Seed-paired vs {base} (Wilcoxon) — worst-group accuracy")
    print("=" * 84)
    for m in ["vanilla", "fairds-1", "ren2018", "fairds-2", "jtt", "groupdro"]:
        if m not in by["test_worst"] or m == base:
            continue
        common = sorted(set(by["test_worst"][base]) & set(by["test_worst"][m]))
        x = np.array([by["test_worst"][base][s] for s in common])
        y = np.array([by["test_worst"][m][s] for s in common])
        try:
            p = wilcoxon(x, y).pvalue
        except ValueError:
            p = float("nan")
        tag = ""
        if m in ("vanilla", "fairds-1", "ren2018"):
            tag = "  PASS" if (x.mean() - y.mean() >= 0.05 or p < 0.05) else "  --"
        print(f"{base} vs {m:12s}: Δworst={x.mean()-y.mean():+.3f}  p={p:.4f}{tag}")


if __name__ == "__main__":
    main()
