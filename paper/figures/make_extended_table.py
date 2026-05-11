"""Generate extended baseline table for the Round 4 paper update."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TBL_DIR = PROJECT_ROOT / "paper" / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    runs = json.load(open(PROJECT_ROOT / "results/e3/20260507-063956/sweep_results.json"))["runs"]
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = r["method"]
        by[m]["val_eval_worst"].append(r["val_eval_worst"])
        by[m]["test_acc"].append(r["test_acc"])
        by[m]["test_worst"].append(r["test_worst"])
        by[m]["seed"].append(r["seed"])

    methods = [
        ("vanilla", "Vanilla", "no", "—"),
        ("fairds-1", "Fairds-1", "no", "closed-form"),
        ("fairds-2", "Fairds-2", "no", "closed-form"),
        ("ren2018", "FORML", "no", "bi-level"),
        ("jtt", "JTT", "no", "2-stage ERM"),
        ("groupdro", "GroupDRO", "yes", "minimax"),
        ("irm", "IRM", "yes", "penalty"),
    ]

    # ---- Extended main table ----
    rows = []
    rows.append(r"\begin{tabular}{llcccc}")
    rows.append(r"\toprule")
    rows.append(r"Method & Family & Group label? & val\_eval\_worst & test\_acc & test\_worst \\")
    rows.append(r"\midrule")
    for key, label, gl, fam in methods:
        d = by[key]
        if not d.get("test_worst"): continue
        is_best = key == "groupdro"   # actual best
        is_2nd_best = key == "jtt"
        is_ours = key in ("fairds-1", "fairds-2")
        cells = [
            f"{np.mean(d['val_eval_worst']):.3f} $\\pm$ {np.std(d['val_eval_worst']):.3f}",
            f"{np.mean(d['test_acc']):.3f} $\\pm$ {np.std(d['test_acc']):.3f}",
            f"{np.mean(d['test_worst']):.3f} $\\pm$ {np.std(d['test_worst']):.3f}",
        ]
        if is_best:
            cells = [r"\textbf{" + c + "}" for c in cells]
            label = r"\textbf{" + label + "}"
        if is_ours:
            label = r"\textit{" + label + "}"
        rows.append(f"{label} & {fam} & {gl} & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (TBL_DIR / "cmnist_extended.tex").write_text("\n".join(rows))

    # ---- Pairwise vs fairds-2 ----
    f2 = np.array(by["fairds-2"]["test_worst"])
    pairs = []
    pairs.append(r"\begin{tabular}{lccc}")
    pairs.append(r"\toprule")
    pairs.append(r"Method (vs Fairds-2) & $\Delta$ test\_worst & $t$ & $p$ (one-sided $>$) \\")
    pairs.append(r"\midrule")
    for key, label, _, _ in methods:
        if key == "fairds-2" or not by[key].get("test_worst"): continue
        a = np.array(by[key]["test_worst"])
        t, p2 = ttest_rel(a, f2)
        p = p2 / 2 if t > 0 else 1 - p2 / 2
        flag = ""
        if p < 0.05 and (a - f2).mean() > 0:
            flag = r" $\uparrow$"
        elif (1 - p) < 0.05:
            flag = r" $\downarrow$"
        pairs.append(f"{label} & {(a-f2).mean():+.4f} & {t:.2f} & {p:.3g}{flag} \\\\")
    pairs.append(r"\bottomrule")
    pairs.append(r"\end{tabular}")
    (TBL_DIR / "cmnist_pairwise_f2.tex").write_text("\n".join(pairs))

    # ---- Bar figure with all 7 methods ----
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    keys = [k for k, *_ in methods]
    means = [np.mean(by[k]["test_worst"]) for k in keys]
    stds = [np.std(by[k]["test_worst"]) for k in keys]
    color_map = {
        "vanilla": "#888888", "fairds-1": "#4C72B0", "fairds-2": "#0F9D58",
        "ren2018": "#DB4437", "jtt": "#7E57C2", "groupdro": "#FF7043", "irm": "#9E9E9E",
    }
    colors = [color_map[k] for k in keys]
    xs = np.arange(len(keys))
    bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=2.5, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in methods], rotation=25, ha="right")
    ax.set_ylabel("OOD test worst-group acc")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.694, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_title("Colored MNIST: extended baseline comparison (10 seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cmnist_extended.pdf")
    plt.close(fig)
    print("[ext] wrote tables and fig_cmnist_extended.pdf")


if __name__ == "__main__":
    main()
