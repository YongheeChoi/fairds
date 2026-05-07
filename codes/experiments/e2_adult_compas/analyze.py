"""Analyze E2 Adult/COMPAS sweep.

Tests:
  - C2 (Fairds vs vanilla on real data fairness metrics): accuracy, DP-diff, EO-diff
  - C4 (sensitive-attribute-free validation): does fairds maintain >= 90% of
    fairness recovery when D_val anchor is unstratified ('random' val_mode)
    compared to the balanced anchor?

Output: SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


def load_runs(p: Path):
    with open(p) as f:
        return json.load(f)["runs"]


def aggregate(runs):
    """method × dataset × val_mode → dict of metric -> list."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in runs:
        m, ds, vm = r["method"], r["dataset"], r["val_mode"]
        for k in ("val_acc", "worst_acc", "dp_diff", "eo_diff", "val_acc_g0", "val_acc_g1", "walltime_sec"):
            v = r.get(k)
            if v is not None:
                out[m][ds][vm][k].append(v)
    return out


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{p}f}"


def report(agg):
    L = []
    methods = sorted(agg.keys())
    datasets = sorted({ds for m in agg for ds in agg[m]})
    val_modes = sorted({vm for m in agg for ds in agg[m] for vm in agg[m][ds]})

    L.append("# E2 — Adult / COMPAS sweep summary\n")
    L.append("## Per-method × dataset × val-mode (mean ± std over seeds)\n")
    L.append("| dataset | method | val_mode | val_acc | worst_acc | dp_diff | eo_diff | walltime |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for m in methods:
            for vm in val_modes:
                d = agg[m][ds][vm]
                if not d.get("val_acc"):
                    continue
                L.append(
                    f"| {ds} | {m} | {vm} "
                    f"| {fmt(np.mean(d['val_acc']))}±{fmt(np.std(d['val_acc']),3)} "
                    f"| {fmt(np.mean(d['worst_acc']))}±{fmt(np.std(d['worst_acc']),3)} "
                    f"| {fmt(np.mean(d['dp_diff']))}±{fmt(np.std(d['dp_diff']),3)} "
                    f"| {fmt(np.mean(d['eo_diff']))}±{fmt(np.std(d['eo_diff']),3)} "
                    f"| {fmt(np.mean(d['walltime_sec']),2)}s |"
                )
    L.append("")

    # vs vanilla, paired t-test
    if "vanilla" in agg:
        L.append("## C2 — Δ vs Vanilla (paired t-test, balanced val_mode)\n")
        L.append("| dataset | method | Δval_acc | p_acc | Δdp_diff | p_dp | Δeo_diff | p_eo |")
        L.append("|---|---|---|---|---|---|---|---|")
        for ds in datasets:
            v_d = agg["vanilla"][ds].get("balanced", {})
            for m in methods:
                if m == "vanilla":
                    continue
                m_d = agg[m][ds].get("balanced", {})
                if not v_d.get("val_acc") or not m_d.get("val_acc") or len(v_d["val_acc"]) != len(m_d["val_acc"]):
                    continue
                row = f"| {ds} | {m} "
                for key in ("val_acc", "dp_diff", "eo_diff"):
                    diff = np.array(m_d[key]) - np.array(v_d[key])
                    t, p_two = ttest_rel(m_d[key], v_d[key])
                    row += f"| {fmt(diff.mean(), 4)} | {fmt(p_two, 4)} "
                row += "|"
                L.append(row)
        L.append("")

    # C4 ablation — balanced vs random
    L.append("## C4 — Sensitive-attribute-free anchor (balanced vs random val_mode)\n")
    L.append("Δ(metric, random − balanced) for each method/dataset. ")
    L.append("If Fairds is robust to lack of sensitive labels, fairness metrics should be similar across val modes.\n")
    L.append("| dataset | method | Δval_acc (rand-bal) | Δdp_diff | Δeo_diff |")
    L.append("|---|---|---|---|---|")
    for ds in datasets:
        for m in methods:
            bal = agg[m][ds].get("balanced", {})
            rnd = agg[m][ds].get("random", {})
            if not bal.get("val_acc") or not rnd.get("val_acc") or len(bal["val_acc"]) != len(rnd["val_acc"]):
                continue
            row = f"| {ds} | {m} "
            for key in ("val_acc", "dp_diff", "eo_diff"):
                d = np.array(rnd[key]) - np.array(bal[key])
                row += f"| {fmt(d.mean(), 4)} "
            row += "|"
            L.append(row)
    L.append("")

    # fairness recovery rate for fairds
    if "vanilla" in agg and "fairds-1" in agg and "fairds-2" in agg:
        L.append("## C4 detail — fairness recovery rate")
        L.append("Recovery rate := (vanilla_metric − fairds_metric) / vanilla_metric, evaluated separately on balanced vs random anchors.\n")
        L.append("Larger positive value = stronger improvement vs vanilla. If fairds-* recovery_random / recovery_balanced ≥ 0.9, C4 is supported.\n")
        L.append("| dataset | method | metric | rec(balanced) | rec(random) | rec_random / rec_balanced |")
        L.append("|---|---|---|---|---|---|")
        for ds in datasets:
            for metric in ("dp_diff", "eo_diff"):
                v_bal = np.mean(agg["vanilla"][ds].get("balanced", {}).get(metric, [np.nan]))
                v_rnd = np.mean(agg["vanilla"][ds].get("random", {}).get(metric, [np.nan]))
                for m in ("fairds-1", "fairds-2"):
                    f_bal = np.mean(agg[m][ds].get("balanced", {}).get(metric, [np.nan]))
                    f_rnd = np.mean(agg[m][ds].get("random", {}).get(metric, [np.nan]))
                    rec_bal = (v_bal - f_bal) / v_bal if v_bal else float("nan")
                    rec_rnd = (v_rnd - f_rnd) / v_rnd if v_rnd else float("nan")
                    ratio = rec_rnd / rec_bal if (rec_bal and not np.isnan(rec_bal) and abs(rec_bal) > 1e-6) else float("nan")
                    L.append(f"| {ds} | {m} | {metric} | {fmt(rec_bal*100, 1)}% | {fmt(rec_rnd*100, 1)}% | {fmt(ratio*100 if not np.isnan(ratio) else float('nan'), 1)}% |")
        L.append("")
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
