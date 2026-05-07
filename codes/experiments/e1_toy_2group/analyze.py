"""Analyze E1 sweep results.

Tests claim C3:
  (a) at majority_ratio = 0.9, fairds-2 should give majority-group phi
      mean STRICTLY BELOW minority-group phi mean (Wilcoxon p < 0.05)
  (b) the (majority - minority) phi gap should DECREASE monotonically
      as imbalance grows (50:50 -> 99:1)

Also reports DP-diff and accuracy by method/ratio and contrasts
fairds-1 vs fairds-2 to isolate the 2nd-order contribution.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d["runs"]


def aggregate(runs: list[dict]) -> dict:
    """Returns nested dict: method -> ratio -> {field: list-over-seeds}."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        m = r["method"]
        ratio = r["majority_ratio"]
        out[m][ratio]["acc"].append(r["final_val_acc"])
        out[m][ratio]["dp_diff"].append(r["dp_diff"])
        out[m][ratio]["eo_diff"].append(r["eo_diff"])
        out[m][ratio]["walltime"].append(r["walltime_sec"])
        # Group-level acc
        for g_id in (0, 1):
            v = r["val_acc_by_group"].get(str(g_id))
            if v is not None:
                out[m][ratio][f"val_acc_g{g_id}"].append(v)
        # Phi summaries (only for fairds methods)
        if r["phi_per_group_last"]:
            for g_id in ("0", "1"):
                stats_g = r["phi_per_group_last"].get(g_id, {})
                if "mean" in stats_g:
                    out[m][ratio][f"phi_mean_g{g_id}"].append(stats_g["mean"])
                    out[m][ratio][f"phi_median_g{g_id}"].append(stats_g["median"])
                    out[m][ratio][f"phi_std_g{g_id}"].append(stats_g["std"])
                    # Per-run pooled values for distribution-level Wilcoxon.
                    out[m][ratio][f"phi_values_g{g_id}"].append(stats_g.get("values", []))
        if r["weight_per_group_last"]:
            for g_id in ("0", "1"):
                stats_g = r["weight_per_group_last"].get(g_id, {})
                if "mean" in stats_g:
                    out[m][ratio][f"w_mean_g{g_id}"].append(stats_g["mean"])
    return out


def fmt(x: float, p: int = 3) -> str:
    return f"{x:.{p}f}"


def summarize(agg: dict) -> str:
    """Pretty-print method × ratio table."""
    lines: list[str] = []
    lines.append("# E1 Sweep Summary (means over seeds)")
    lines.append("")

    methods = sorted(agg.keys())
    ratios = sorted({r for m in agg for r in agg[m]})

    lines.append("## Validation accuracy / DP-diff / EO-diff")
    header = "| method | ratio | val_acc | val_acc_g0 (maj) | val_acc_g1 (min) | dp_diff | eo_diff | walltime/run |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        for r in ratios:
            d = agg[m][r]
            row = (
                f"| {m} | {r:.2f} "
                f"| {fmt(np.mean(d['acc']))} "
                f"| {fmt(np.mean(d.get('val_acc_g0', [float('nan')])))} "
                f"| {fmt(np.mean(d.get('val_acc_g1', [float('nan')])))} "
                f"| {fmt(np.mean(d['dp_diff']))} "
                f"| {fmt(np.mean(d['eo_diff']))} "
                f"| {fmt(np.mean(d['walltime']), 2)}s |"
            )
            lines.append(row)
    lines.append("")

    lines.append("## Per-group Shapley value (mean across seeds)")
    lines.append("Δφ = mean(φ | majority) − mean(φ | minority).")
    lines.append("If C3 holds (representation-bias attenuation), Δφ should decrease (preferably go negative) as imbalance grows for fairds-2.")
    lines.append("")
    header2 = "| method | ratio | mean_phi_majority | mean_phi_minority | Δφ (maj − min) | mean_w_majority | mean_w_minority | Δw |"
    sep2 = "|---|---|---|---|---|---|---|---|"
    lines.append(header2)
    lines.append(sep2)
    for m in methods:
        if not m.startswith("fairds"):
            continue
        for r in ratios:
            d = agg[m][r]
            phi_maj = np.mean(d.get("phi_mean_g0", [float("nan")]))
            phi_min = np.mean(d.get("phi_mean_g1", [float("nan")]))
            dphi = phi_maj - phi_min
            w_maj = np.mean(d.get("w_mean_g0", [float("nan")]))
            w_min = np.mean(d.get("w_mean_g1", [float("nan")]))
            dw = w_maj - w_min
            lines.append(
                f"| {m} | {r:.2f} | {fmt(phi_maj, 5)} | {fmt(phi_min, 5)} | {fmt(dphi, 5)} | {fmt(w_maj, 4)} | {fmt(w_min, 4)} | {fmt(dw, 4)} |"
            )
    lines.append("")

    lines.append("## C3 hypothesis tests")
    lines.append("- (a) For majority_ratio=0.90, test fairds-2 majority-vs-minority phi via Mann–Whitney U on the pooled per-sample values across all 5 seeds.")
    lines.append("- (b) Monotone decrease of Δφ with ratio for fairds-2.")
    lines.append("")

    target_method = "fairds-2"
    if target_method in agg:
        # (a) Mann-Whitney on pooled raw values at ratio=0.9
        if 0.9 in agg[target_method]:
            d = agg[target_method][0.9]
            maj_vals = []
            min_vals = []
            for seed_list in d.get("phi_values_g0", []):
                maj_vals.extend(seed_list)
            for seed_list in d.get("phi_values_g1", []):
                min_vals.extend(seed_list)
            if maj_vals and min_vals:
                stat, p = mannwhitneyu(maj_vals, min_vals, alternative="less")
                lines.append(
                    f"- (a) ratio=0.90, Mann–Whitney U(maj < min) for {target_method}: "
                    f"U={stat:.0f}, p={p:.4g}, n_maj={len(maj_vals)}, n_min={len(min_vals)}"
                )
                lines.append(f"  → **C3(a) {'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}** at α=0.05")

        # (b) Monotonicity of Δφ
        deltas = []
        for r in sorted(agg[target_method].keys()):
            d = agg[target_method][r]
            phi_maj = np.mean(d.get("phi_mean_g0", [float("nan")]))
            phi_min = np.mean(d.get("phi_mean_g1", [float("nan")]))
            deltas.append((r, phi_maj - phi_min))
        diffs = [d2 - d1 for (_, d1), (_, d2) in zip(deltas, deltas[1:])]
        monotone_decreasing = all(d <= 0 for d in diffs) and any(d < 0 for d in diffs)
        lines.append(f"- (b) Δφ across ratios for {target_method}: " + ", ".join(f"{r:.2f}→{d:+.5f}" for r, d in deltas))
        lines.append(f"  → **C3(b) {'SUPPORTED (monotone decreasing)' if monotone_decreasing else 'NOT SUPPORTED'}**")
    lines.append("")

    lines.append("## fairds-1 vs fairds-2: isolation of the 2nd-order contribution")
    if "fairds-1" in agg and "fairds-2" in agg:
        lines.append("Compare Δφ for fairds-1 vs fairds-2 across ratios:")
        lines.append("| ratio | Δφ (fairds-1) | Δφ (fairds-2) | Δφ₂ − Δφ₁ |")
        lines.append("|---|---|---|---|")
        for r in ratios:
            d1 = agg["fairds-1"][r]
            d2 = agg["fairds-2"][r]
            phi_maj1 = np.mean(d1.get("phi_mean_g0", [float("nan")]))
            phi_min1 = np.mean(d1.get("phi_mean_g1", [float("nan")]))
            phi_maj2 = np.mean(d2.get("phi_mean_g0", [float("nan")]))
            phi_min2 = np.mean(d2.get("phi_mean_g1", [float("nan")]))
            delta1 = phi_maj1 - phi_min1
            delta2 = phi_maj2 - phi_min2
            lines.append(f"| {r:.2f} | {delta1:+.5f} | {delta2:+.5f} | {delta2 - delta1:+.5f} |")
    lines.append("")

    lines.append("## Wall-clock (rough overhead vs vanilla)")
    if "vanilla" in agg:
        for r in ratios:
            v = np.mean(agg["vanilla"][r]["walltime"])
            row = f"- ratio={r:.2f}: vanilla={v:.2f}s "
            for m in ("fairds-1", "fairds-2"):
                if m in agg:
                    fm = np.mean(agg[m][r]["walltime"])
                    row += f"| {m}={fm:.2f}s ({fm / v:.2f}×)"
            lines.append(row)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    runs = load_runs(Path(args.results_path))
    agg = aggregate(runs)
    summary = summarize(agg)

    if args.out:
        Path(args.out).write_text(summary)
        print(f"[analyze] wrote {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
