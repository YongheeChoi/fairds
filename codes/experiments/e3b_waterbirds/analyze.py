"""Analyze Waterbirds sweep — paired t-tests, per-method aggregates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


def load(p):
    with open(p) as f:
        return json.load(f)["runs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    a = args

    runs = load(Path(a.results_path))
    by_m = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = r["method"]
        by_m[m]["val_eval_acc"].append(r["val_eval_acc"])
        by_m[m]["val_eval_worst"].append(r["val_eval_worst"])
        by_m[m]["test_acc"].append(r["test_acc"])
        by_m[m]["test_worst"].append(r["test_worst"])
        by_m[m]["seed"].append(r["seed"])

    L = []
    L.append("# E3b — Waterbirds sweep summary\n")
    L.append("Real-image spurious-correlation benchmark (Sagawa et al. 2020).")
    L.append("Held-out worst-group acc on val_eval (anchor-disjoint) and test (true OOD setting w/ 4 group breakdown).\n")

    L.append("## Per-method aggregates (mean ± std over seeds)\n")
    L.append("| method | val_eval_worst | test_acc | test_worst |")
    L.append("|---|---|---|---|")
    for m in ("vanilla", "fairds-1", "fairds-2", "ren2018"):
        d = by_m[m]
        if not d.get("test_worst"):
            continue
        L.append(
            f"| {m} | {np.mean(d['val_eval_worst']):.3f}±{np.std(d['val_eval_worst']):.3f} "
            f"| {np.mean(d['test_acc']):.3f}±{np.std(d['test_acc']):.3f} "
            f"| {np.mean(d['test_worst']):.3f}±{np.std(d['test_worst']):.3f} |"
        )
    L.append("")

    L.append("## Paired tests vs vanilla (one-sided > 0)\n")
    L.append("| method | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |")
    L.append("|---|---|---|---|---|---|---|")
    if "vanilla" in by_m:
        v_w = np.array(by_m["vanilla"]["val_eval_worst"])
        v_t = np.array(by_m["vanilla"]["test_acc"])
        v_tw = np.array(by_m["vanilla"]["test_worst"])
        for m in ("fairds-1", "fairds-2", "ren2018"):
            mw = np.array(by_m[m]["val_eval_worst"])
            mt = np.array(by_m[m]["test_acc"])
            mtw = np.array(by_m[m]["test_worst"])
            row = f"| {m} "
            for arr_x, arr_y in [(mw, v_w), (mt, v_t), (mtw, v_tw)]:
                if len(arr_x) != len(arr_y) or len(arr_x) < 2:
                    row += "| — | — "
                    continue
                t, p2 = ttest_rel(arr_x, arr_y)
                p = p2 / 2 if t > 0 else 1 - p2 / 2
                row += f"| {(arr_x-arr_y).mean():+.4f} | {p:.4g} "
            row += "|"
            L.append(row)
    L.append("")

    L.append("## Isolation tests\n")
    L.append("| comparison | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |")
    L.append("|---|---|---|---|---|---|---|")
    for ref in ("fairds-1", "ren2018"):
        if "fairds-2" not in by_m or ref not in by_m:
            continue
        f2_w = np.array(by_m["fairds-2"]["val_eval_worst"])
        f2_t = np.array(by_m["fairds-2"]["test_acc"])
        f2_tw = np.array(by_m["fairds-2"]["test_worst"])
        r_w = np.array(by_m[ref]["val_eval_worst"])
        r_t = np.array(by_m[ref]["test_acc"])
        r_tw = np.array(by_m[ref]["test_worst"])
        row = f"| fairds-2 vs {ref} "
        for arr_x, arr_y in [(f2_w, r_w), (f2_t, r_t), (f2_tw, r_tw)]:
            if len(arr_x) != len(arr_y) or len(arr_x) < 2:
                row += "| — | — "
                continue
            t, p2 = ttest_rel(arr_x, arr_y)
            p = p2 / 2 if t > 0 else 1 - p2 / 2
            row += f"| {(arr_x-arr_y).mean():+.4f} | {p:.4g} "
        row += "|"
        L.append(row)
    L.append("")

    L.append("## Per-seed test_worst details\n")
    L.append("| seed | vanilla | fairds-1 | fairds-2 | ren2018 |")
    L.append("|---|---|---|---|---|")
    seeds = sorted(set(by_m["vanilla"]["seed"])) if "vanilla" in by_m else []
    for s in seeds:
        row = f"| {s} "
        for m in ("vanilla", "fairds-1", "fairds-2", "ren2018"):
            d = by_m[m]
            try:
                idx = d["seed"].index(s)
                row += f"| {d['test_worst'][idx]:.3f} "
            except (ValueError, KeyError):
                row += "| — "
        row += "|"
        L.append(row)
    L.append("")

    txt = "\n".join(L)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"[analyze] wrote {a.out}")
    print(txt)


if __name__ == "__main__":
    main()
