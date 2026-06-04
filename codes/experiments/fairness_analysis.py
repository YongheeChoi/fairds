"""Fairness re-analysis (group fairness lens).

For each spurious/biased regime we report, per method:
  - worst-group accuracy  (= Rawlsian / max-min fairness; higher = fairer)
  - accuracy disparity gap = acc(spurious-aligned group) - acc(flipped group)
        (= how much the model favors the biased majority; lower = fairer)
  - overall accuracy

This recasts the existing results as a fairness story: does the closed-form
2nd-order Shapley reweighter reduce the gap between the biased-majority group and
the under-represented (flipped) group, while keeping accuracy?
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("/home/users/yonghee/projects/fairds")


def latest(*patterns):
    out = []
    for pat in patterns:
        fs = sorted(glob.glob(str(pat)))
        if fs:
            out.append(fs[-1])
    return out


REGIMES = {
    "CMNIST (color)": latest(BASE / "results/e3/20260507-063956/sweep_results.json"),
    "Corrupted-CIFAR (texture)": latest(BASE / "results/e5_corrupted_cifar/*/sweep_results.json"),
    "Spurious-STL10 (natural)": latest(
        BASE / "codes/results/e6_main/*/sweep_results.json",
        BASE / "codes/results/e6_resid/*/sweep_results.json",
        BASE / "codes/results/e6_oracle/*/sweep_results.json",
    ),
}

ORDER = ["vanilla", "fairds-1", "fairds-2", "fairds-2-resid", "ren2018", "jtt", "groupdro"]


def load(p):
    return json.load(open(p))["runs"]


def main():
    for regime, paths in REGIMES.items():
        runs = []
        for p in paths:
            runs += load(p)
        by = defaultdict(list)
        for r in runs:
            key = r["method"] + ("-resid" if r.get("arm") == "residual_real" else "")
            a = r.get("test_acc_aligned")
            f = r.get("test_acc_flipped")
            if a is None or f is None:
                continue
            by[key].append((r["test_worst"], r["test_acc"], a - f))
        print(f"\n=== {regime} ===")
        print(f"{'method':16s} {'worst-grp':>9s} {'disparity':>10s} {'overall':>8s}")
        keys = [k for k in ORDER if k in by] + [k for k in by if k not in ORDER]
        for k in keys:
            arr = np.array(by[k])
            print(f"{k:16s} {arr[:,0].mean():>9.3f} {arr[:,2].mean():>10.3f} {arr[:,1].mean():>8.3f}")

    # E2 demographic fairness (from SUMMARY, balanced val_mode) for reference
    print("\n=== Adult/COMPAS demographic (DP / EO; lower = fairer) ===")
    e2 = {
        "adult": {"vanilla": (0.196, 0.117, 0.837), "fairds-1": (0.185, 0.127, 0.835),
                  "fairds-2": (0.189, 0.127, 0.835), "ren2018": (0.175, 0.102, 0.819)},
        "compas": {"vanilla": (0.240, 0.226, 0.667), "fairds-1": (0.238, 0.240, 0.667),
                   "fairds-2": (0.239, 0.244, 0.671), "ren2018": (0.231, 0.241, 0.681)},
    }
    for ds, d in e2.items():
        print(f"-- {ds} --   {'method':10s} {'DP':>6s} {'EO':>6s} {'acc':>6s}")
        for m, (dp, eo, acc) in d.items():
            print(f"{'':14s}{m:10s} {dp:>6.3f} {eo:>6.3f} {acc:>6.3f}")


if __name__ == "__main__":
    main()
