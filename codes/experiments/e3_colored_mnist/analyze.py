"""Analyze E3 Colored MNIST sweep — paired t-tests and per-method aggregates."""

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

    runs = load(Path(args.results_path))
    by_method = defaultdict(lambda: defaultdict(list))
    # Field names changed in Round 5: val_acc → val_eval_acc, val_worst → val_eval_worst,
    # plus test_worst (held-out OOD per-group). Fall back gracefully for old runs.
    val_acc_key = "val_eval_acc" if any("val_eval_acc" in r for r in runs) else "val_acc"
    val_worst_key = "val_eval_worst" if any("val_eval_worst" in r for r in runs) else "val_worst"
    has_test_worst = any("test_worst" in r for r in runs)

    for r in runs:
        m = r["method"]
        by_method[m]["val_acc"].append(r.get(val_acc_key))
        v_w = r.get(val_worst_key)
        if v_w is not None:
            by_method[m]["val_worst"].append(v_w)
        by_method[m]["test_acc"].append(r["test_acc"])
        if has_test_worst and r.get("test_worst") is not None:
            by_method[m]["test_worst"].append(r["test_worst"])
        by_method[m]["seed"].append(r["seed"])

    L = []
    L.append("# E3 — Colored MNIST sweep summary (post Round 3 fix)\n")
    L.append("Standard spurious-correlation benchmark per Codex Round 3 review.\n")
    L.append("Setup: 5000 train samples, ratio=0.9, 10 seeds, 20 epochs, τ=0.1, ws=4.0.\n")

    L.append("## Per-method aggregates (mean ± std over 10 seeds, held-out)\n")
    if has_test_worst:
        L.append("| method | val_eval_worst | test_acc | test_worst |")
        L.append("|---|---|---|---|")
        for m in ("vanilla", "fairds-1", "fairds-2", "ren2018"):
            d = by_method[m]
            L.append(
                f"| {m} | {np.mean(d['val_worst']):.3f}±{np.std(d['val_worst']):.3f} "
                f"| {np.mean(d['test_acc']):.3f}±{np.std(d['test_acc']):.3f} "
                f"| {np.mean(d['test_worst']):.3f}±{np.std(d['test_worst']):.3f} |"
            )
    else:
        L.append("| method | val_acc | val_worst | test_acc |")
        L.append("|---|---|---|---|")
        for m in ("vanilla", "fairds-1", "fairds-2", "ren2018"):
            d = by_method[m]
            L.append(
                f"| {m} | {np.mean(d['val_acc']):.3f}±{np.std(d['val_acc']):.3f} "
                f"| {np.mean(d['val_worst']):.3f}±{np.std(d['val_worst']):.3f} "
                f"| {np.mean(d['test_acc']):.3f}±{np.std(d['test_acc']):.3f} |"
            )
    L.append("")

    L.append("## Paired tests (each method vs vanilla, same seeds)\n")
    if has_test_worst:
        L.append("| method | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |")
        L.append("|---|---|---|---|---|---|---|")
        v_w = np.array(by_method["vanilla"]["val_worst"])
        v_t = np.array(by_method["vanilla"]["test_acc"])
        v_tw = np.array(by_method["vanilla"]["test_worst"])
        for m in ("fairds-1", "fairds-2", "ren2018"):
            mw = np.array(by_method[m]["val_worst"])
            mt = np.array(by_method[m]["test_acc"])
            mtw = np.array(by_method[m]["test_worst"])
            row = f"| {m} "
            for a, b in [(mw, v_w), (mt, v_t), (mtw, v_tw)]:
                t, p2 = ttest_rel(a, b)
                p = p2 / 2 if t > 0 else 1 - p2 / 2
                row += f"| {(a-b).mean():+.4f} | {p:.4g} "
            row += "|"
            L.append(row)
    else:
        L.append("| method | Δval_worst | t (paired) | p | Δtest_acc | t | p |")
        L.append("|---|---|---|---|---|---|---|")
        v_w = np.array(by_method["vanilla"]["val_worst"])
        v_t = np.array(by_method["vanilla"]["test_acc"])
        for m in ("fairds-1", "fairds-2", "ren2018"):
            mw = np.array(by_method[m]["val_worst"])
            mt = np.array(by_method[m]["test_acc"])
            t_w, p_w_two = ttest_rel(mw, v_w)
            p_w = p_w_two / 2 if t_w > 0 else 1 - p_w_two / 2
            t_t, p_t_two = ttest_rel(mt, v_t)
            p_t = p_t_two / 2 if t_t > 0 else 1 - p_t_two / 2
            L.append(
                f"| {m} | {(mw - v_w).mean():+.4f} | {t_w:.2f} | {p_w:.4g} "
                f"| {(mt - v_t).mean():+.4f} | {t_t:.2f} | {p_t:.4g} |"
            )
    L.append("")

    # Isolation: fairds-2 vs fairds-1 and vs ren2018
    L.append("## Isolation tests (fairds-2 vs fairds-1, fairds-2 vs ren2018)\n")
    if has_test_worst:
        L.append("| comparison | Δval_eval_worst | p | Δtest_acc | p | Δtest_worst | p |")
        L.append("|---|---|---|---|---|---|---|")
        f2_w = np.array(by_method["fairds-2"]["val_worst"])
        f2_t = np.array(by_method["fairds-2"]["test_acc"])
        f2_tw = np.array(by_method["fairds-2"]["test_worst"])
        for ref in ("fairds-1", "ren2018"):
            r_w = np.array(by_method[ref]["val_worst"])
            r_t = np.array(by_method[ref]["test_acc"])
            r_tw = np.array(by_method[ref]["test_worst"])
            row = f"| fairds-2 vs {ref} "
            for a, b in [(f2_w, r_w), (f2_t, r_t), (f2_tw, r_tw)]:
                t, p2 = ttest_rel(a, b)
                p = p2 / 2 if t > 0 else 1 - p2 / 2
                row += f"| {(a-b).mean():+.4f} | {p:.4g} "
            row += "|"
            L.append(row)
    else:
        L.append("| comparison | Δval_worst | t | p | Δtest_acc | t | p |")
        L.append("|---|---|---|---|---|---|---|")
        f2_w = np.array(by_method["fairds-2"]["val_worst"])
        f2_t = np.array(by_method["fairds-2"]["test_acc"])
        for ref in ("fairds-1", "ren2018"):
            r_w = np.array(by_method[ref]["val_worst"])
            r_t = np.array(by_method[ref]["test_acc"])
            t_w, p_w_two = ttest_rel(f2_w, r_w)
            p_w = p_w_two / 2 if t_w > 0 else 1 - p_w_two / 2
            t_t, p_t_two = ttest_rel(f2_t, r_t)
            p_t = p_t_two / 2 if t_t > 0 else 1 - p_t_two / 2
            L.append(
                f"| fairds-2 vs {ref} | {(f2_w - r_w).mean():+.4f} | {t_w:.2f} | {p_w:.4g} "
                f"| {(f2_t - r_t).mean():+.4f} | {t_t:.2f} | {p_t:.4g} |"
            )
    L.append("")

    L.append("## Per-seed details (val_worst)\n")
    L.append("| seed | vanilla | fairds-1 | fairds-2 | ren2018 |")
    L.append("|---|---|---|---|---|")
    seeds = sorted(set(by_method["vanilla"]["seed"]))
    for s in seeds:
        row = f"| {s} "
        for m in ("vanilla", "fairds-1", "fairds-2", "ren2018"):
            d = by_method[m]
            try:
                idx = d["seed"].index(s)
                row += f"| {d['val_worst'][idx]:.3f} "
            except (ValueError, KeyError):
                row += "| — "
        row += "|"
        L.append(row)
    L.append("")

    txt = "\n".join(L)
    if args.out:
        Path(args.out).write_text(txt)
        print(f"[analyze] wrote {args.out}")
    print(txt)


if __name__ == "__main__":
    main()
