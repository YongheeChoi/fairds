"""Analyze E1b sweep — spurious-feature MLP.

Tests claim C3 in a setup where vanilla MLP is *forced* to use a
spurious feature held in the majority group. Worst-group accuracy is
the primary metric: if fairds-2 attenuates spurious shortcut, minority
acc should rise above vanilla's.

Outputs SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, ttest_rel


def load_runs(p: Path):
    with open(p) as f:
        return json.load(f)["runs"]


def aggregate(runs):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        m = r["method"]
        ratio = r["majority_ratio"]
        out[m][ratio]["val_acc"].append(r["final_val_acc"])
        out[m][ratio]["val_acc_worst"].append(r["final_val_acc_worst"])
        out[m][ratio]["dp_diff"].append(r["dp_diff"])
        out[m][ratio]["eo_diff"].append(r["eo_diff"])
        for gid in (0, 1):
            v = r["val_acc_by_group"].get(str(gid))
            if v is not None:
                out[m][ratio][f"val_acc_g{gid}"].append(v)
        if r.get("phi_per_group_last"):
            for gid in ("0", "1"):
                stats_g = r["phi_per_group_last"].get(gid, {})
                if "mean" in stats_g:
                    out[m][ratio][f"phi_mean_g{gid}"].append(stats_g["mean"])
                    out[m][ratio][f"phi_values_g{gid}"].append(stats_g.get("values", []))
        if r.get("weight_per_group_last"):
            for gid in ("0", "1"):
                stats_g = r["weight_per_group_last"].get(gid, {})
                if "mean" in stats_g:
                    out[m][ratio][f"w_mean_g{gid}"].append(stats_g["mean"])
        out[m][ratio]["walltime"].append(r.get("walltime_sec", float("nan")))
    return out


def fmt(x, p=3):
    return f"{x:.{p}f}"


def report(agg):
    L = []
    methods = sorted(agg.keys())
    ratios = sorted({r for m in agg for r in agg[m]})
    L.append("# E1b — Spurious-feature 2-group MLP sweep summary\n")

    L.append("## Worst-group accuracy (primary metric)\n")
    L.append("| method | ratio | val_acc | worst | val_acc_g0 (maj) | val_acc_g1 (min) | dp_diff | eo_diff |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in methods:
        for r in ratios:
            d = agg[m][r]
            L.append(
                f"| {m} | {r:.2f} "
                f"| {fmt(np.mean(d['val_acc']))} "
                f"| {fmt(np.mean(d['val_acc_worst']))} "
                f"| {fmt(np.mean(d.get('val_acc_g0', [float('nan')])))} "
                f"| {fmt(np.mean(d.get('val_acc_g1', [float('nan')])))} "
                f"| {fmt(np.mean(d['dp_diff']))} "
                f"| {fmt(np.mean(d['eo_diff']))} |"
            )
    L.append("")

    # Δworst-group acc vs vanilla, paired t-test across seeds
    if "vanilla" in agg:
        L.append("## Δworst (method − vanilla), paired across seeds\n")
        L.append("| method | ratio | mean Δworst | t-stat | p (one-sided > 0) |")
        L.append("|---|---|---|---|---|")
        for m in methods:
            if m == "vanilla":
                continue
            for r in ratios:
                v_worst = agg["vanilla"][r]["val_acc_worst"]
                m_worst = agg[m][r]["val_acc_worst"]
                if len(v_worst) == len(m_worst) and len(v_worst) > 1:
                    diffs = np.array(m_worst) - np.array(v_worst)
                    t, p_two = ttest_rel(m_worst, v_worst)
                    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
                    L.append(f"| {m} | {r:.2f} | {fmt(diffs.mean(), 4)} | {fmt(t, 3)} | {fmt(p_one, 4)} |")
        L.append("")

    # Δworst between fairds-1 and fairds-2 (isolation of 2nd-order)
    if "fairds-1" in agg and "fairds-2" in agg:
        L.append("## Isolation of 2nd-order: Δworst (fairds-2 − fairds-1)\n")
        L.append("| ratio | fairds-1 worst | fairds-2 worst | Δ |")
        L.append("|---|---|---|---|")
        for r in ratios:
            w1 = np.mean(agg["fairds-1"][r]["val_acc_worst"])
            w2 = np.mean(agg["fairds-2"][r]["val_acc_worst"])
            L.append(f"| {r:.2f} | {fmt(w1)} | {fmt(w2)} | {fmt(w2 - w1, 4)} |")
        L.append("")

    L.append("## Per-group Shapley value (mean across seeds)")
    L.append("Δφ = mean(φ | majority) − mean(φ | minority); negative = majority gets DOWN-weighted (C3 expected).\n")
    L.append("| method | ratio | mean_phi_maj | mean_phi_min | Δφ | mean_w_maj | mean_w_min | Δw |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in methods:
        if not m.startswith("fairds"):
            continue
        for r in ratios:
            d = agg[m][r]
            phi_maj = np.mean(d.get("phi_mean_g0", [float("nan")]))
            phi_min = np.mean(d.get("phi_mean_g1", [float("nan")]))
            w_maj = np.mean(d.get("w_mean_g0", [float("nan")]))
            w_min = np.mean(d.get("w_mean_g1", [float("nan")]))
            L.append(
                f"| {m} | {r:.2f} | {fmt(phi_maj, 5)} | {fmt(phi_min, 5)} | {fmt(phi_maj - phi_min, 5)} | "
                f"{fmt(w_maj, 4)} | {fmt(w_min, 4)} | {fmt(w_maj - w_min, 4)} |"
            )
    L.append("")

    # Pooled Mann-Whitney for fairds-2 at ratio=0.9 (legacy C3 test)
    if "fairds-2" in agg and 0.9 in agg["fairds-2"]:
        d = agg["fairds-2"][0.9]
        majs = []
        mins = []
        for s in d.get("phi_values_g0", []):
            majs.extend(s)
        for s in d.get("phi_values_g1", []):
            mins.extend(s)
        if majs and mins:
            stat, p = mannwhitneyu(majs, mins, alternative="less")
            L.append("## C3 hypothesis test (Mann-Whitney U, fairds-2, ratio=0.9)\n")
            L.append(f"- U={stat:.0f}, p (one-sided maj < min)={p:.4g}, n_maj={len(majs)}, n_min={len(mins)}")
            L.append(f"- → C3(a) {'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'} at α=0.05\n")

    L.append("## Wall-clock per-run (mean over seeds, ratio=0.9)\n")
    for m in methods:
        if 0.9 in agg[m]:
            L.append(f"- {m}: {np.mean(agg[m][0.9]['walltime']):.2f}s")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_path", type=str)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    runs = load_runs(Path(a.results_path))
    agg = aggregate(runs)
    txt = report(agg)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"[analyze] wrote {a.out}")
    print(txt)


if __name__ == "__main__":
    main()
