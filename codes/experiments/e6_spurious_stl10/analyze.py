"""Analyze E6 Spurious-STL10 results: per-method test_worst, seed-paired stats.

8-clinch criterion (Codex R3): fairds-2 (residual_real) must beat ERM, fairds-1,
ren2018 by >=5pp test_worst (or p<0.05 seed-paired), match jtt/dfr within uncertainty,
may lose to groupdro oracle (reported plainly).
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOTS = ["e6_main", "e6_oracle", "e6_resid"]


def load(root):
    base = Path(__file__).resolve().parents[2]  # codes/
    js = sorted(glob.glob(str(base / f"results/{root}/*/sweep_results.json")))
    return json.load(open(js[-1]))["runs"] if js else []


def main():
    runs = []
    for r in ROOTS:
        runs += load(r)

    by_key = defaultdict(dict)        # key -> {seed: test_worst}
    by_key_val = defaultdict(dict)    # key -> {seed: val_eval_worst}
    by_key_acc = defaultdict(dict)    # key -> {seed: test_acc}
    for r in runs:
        key = r["method"] + ("-resid" if r.get("arm") == "residual_real" else "")
        by_key[key][r["seed"]] = r["test_worst"]
        by_key_val[key][r["seed"]] = r.get("val_eval_worst")
        by_key_acc[key][r["seed"]] = r.get("test_acc")

    order = ["vanilla", "fairds-1", "fairds-2", "fairds-2-resid", "ren2018", "jtt", "groupdro"]
    keys = [k for k in order if k in by_key] + [k for k in by_key if k not in order]

    print("=" * 72)
    print("E6 Spurious-STL10 (car vs truck, 96px, texture-corruption spurious)")
    print("OOD test_worst (spurious-flipped worst group), 10 seeds")
    print("=" * 72)
    print(f"{'method':18s} {'n':>3s} {'test_worst':>18s} {'test_acc':>10s} {'val_worst':>10s}")
    for k in keys:
        a = np.array([by_key[k][s] for s in sorted(by_key[k])])
        acc = np.array([by_key_acc[k][s] for s in sorted(by_key_acc[k])])
        val = np.array([v for v in (by_key_val[k][s] for s in sorted(by_key_val[k])) if v is not None])
        print(f"{k:18s} {len(a):>3d} {a.mean():>8.3f} ± {a.std():>6.3f} "
              f"{acc.mean():>10.3f} {val.mean():>10.3f}")

    # Seed-paired comparisons vs the headline method (fairds-2-resid if present else fairds-2)
    base = "fairds-2-resid" if "fairds-2-resid" in by_key else "fairds-2"
    print("\n" + "=" * 72)
    print(f"Seed-paired comparisons vs {base} (Wilcoxon signed-rank)")
    print("=" * 72)
    for k in ["vanilla", "fairds-1", "ren2018", "fairds-2", "jtt", "groupdro"]:
        if k not in by_key or k == base:
            continue
        common = sorted(set(by_key[base]) & set(by_key[k]))
        x = np.array([by_key[base][s] for s in common])
        y = np.array([by_key[k][s] for s in common])
        delta = x.mean() - y.mean()
        try:
            p = wilcoxon(x, y).pvalue
        except ValueError:
            p = float("nan")
        verdict = ""
        if k in ("vanilla", "fairds-1", "ren2018"):
            verdict = "  <-- 8-clinch target" + (" PASS" if (delta >= 0.05 or p < 0.05) else " FAIL")
        print(f"{base} vs {k:12s}: Δ={delta:+.3f}  paired p={p:.4f}{verdict}")

    # also fairds-2 vs fairds-1 (2nd-order isolation) if base is resid
    if base == "fairds-2-resid" and "fairds-2" in by_key:
        common = sorted(set(by_key["fairds-2-resid"]) & set(by_key["fairds-2"]))
        x = np.array([by_key["fairds-2-resid"][s] for s in common])
        y = np.array([by_key["fairds-2"][s] for s in common])
        try:
            p = wilcoxon(x, y).pvalue
        except ValueError:
            p = float("nan")
        print(f"\n[mechanism] residual_real vs full fairds-2: Δ={x.mean()-y.mean():+.3f} p={p:.4f} "
              f"(preserve => residual carries the 2nd-order signal)")


if __name__ == "__main__":
    main()
