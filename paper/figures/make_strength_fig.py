"""Spurious-strength phase-transition figure for paper appendix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

COLORS = {
    "vanilla": "#888888", "fairds-1": "#4C72B0",
    "fairds-2": "#0F9D58", "ren2018": "#DB4437",
}
LABELS = {
    "vanilla": "Vanilla", "fairds-1": "Fairds-1",
    "fairds-2": "Fairds-2", "ren2018": "FORML",
}


def main():
    runs = json.load(open(PROJECT_ROOT / "results/e3_strength/20260507-064949/sweep_results.json"))["runs"]
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by[(r["method"], r["p_spurious"])]["test_worst"].append(r["test_worst"])

    strengths = [0.7, 0.8, 0.9, 0.95, 0.99]
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    for m in ("vanilla", "fairds-1", "fairds-2", "ren2018"):
        means = [np.mean(by[(m, s)]["test_worst"]) for s in strengths]
        stds = [np.std(by[(m, s)]["test_worst"]) for s in strengths]
        ax.errorbar(strengths, means, yerr=stds,
                    label=LABELS[m], color=COLORS[m],
                    marker="o", markersize=4, linewidth=1.5, capsize=3)
    ax.set_xlabel(r"Spurious correlation strength $p_{\mathrm{maj}}$")
    ax.set_ylabel("OOD test worst-group acc")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(strengths)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Phase transition: Fairds-2 robust as spurious strength grows", fontsize=9)
    ax.axvline(0.9, color="black", linewidth=0.3, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_strength_phase.pdf")
    plt.close(fig)
    print("[fig] wrote fig_strength_phase.pdf")


if __name__ == "__main__":
    main()
