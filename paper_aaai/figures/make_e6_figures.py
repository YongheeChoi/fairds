"""Figures for the fairness paper (Data-Shapley reweighting against data bias).

Frame: a closed-form 2nd-order Data Shapley score down-weights the biased-majority
samples during training, improving GROUP FAIRNESS. We show it across three bias
regimes (color / texture / natural image).

Saves both PDF (LaTeX) and PNG (HTML slides). Numbers are confirmed 10-seed
test results (worst-group) and re-computed group accuracy gaps (disparity).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11})

# ---- worst-group accuracy (= Rawlsian / max-min group fairness; higher = fairer) ----
METHODS7 = ["vanilla", "fairds-1", "fairds-2", "fairds-2\nresidual", "ren2018", "jtt", "groupdro"]
WORST = {
    "Colored MNIST\n(color bias)":         [0.694, 0.777, 0.824, 0.815, 0.796, 0.878, 0.900],
    "Corrupted-CIFAR\n(texture bias)":     [0.196, 0.374, 0.423, 0.458, 0.375, 0.368, 0.564],
    "Spurious-STL10\n(natural-image bias)":[0.042, 0.194, 0.227, 0.323, 0.151, 0.606, 0.521],
}
STL_STD = [0.026, 0.138, 0.154, 0.111, 0.124, 0.043, 0.043]
COL7 = ["#9e9e9e", "#90caf9", "#42a5f5", "#1565c0", "#bdbdbd", "#ef9a9a", "#c62828"]

# ---- group accuracy gap = acc(biased-majority group) - acc(minority group); lower = fairer ----
METHODS6 = ["vanilla", "fairds-1", "fairds-2", "ren2018", "jtt", "groupdro"]
DISPARITY = {
    "Colored MNIST":    [0.223, 0.143, 0.118, 0.169, 0.086, 0.058],
    "Corrupted-CIFAR":  [0.703, 0.512, 0.430, 0.520, 0.431, 0.193],
    "Spurious-STL10":   [0.950, 0.736, 0.698, 0.797, 0.144, 0.259],
}
COL6 = ["#9e9e9e", "#90caf9", "#42a5f5", "#bdbdbd", "#ef9a9a", "#c62828"]


def fig_three_regimes():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    x = np.arange(len(METHODS7))
    for ax, (name, vals) in zip(axes, WORST.items()):
        yerr = STL_STD if "STL" in name else None
        bars = ax.bar(x, vals, color=COL7, yerr=yerr, capsize=2, edgecolor="black", linewidth=0.4)
        bars[3].set_hatch("///")
        ax.set_title(name, fontsize=10.5)
        ax.set_xticks(x); ax.set_xticklabels(METHODS7, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("worst-group accuracy\n(↑ = fairer)")
    fig.suptitle("Group fairness across three data-bias regimes: the 2nd-order Shapley reweighter protects "
                 "the under-represented group\n(grey/blue = no group label; red = uses group label or 2-stage)",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_three_regimes.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig_three_regimes.{pdf,png}")


def fig_disparity():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    x = np.arange(len(METHODS6))
    for ax, (name, vals) in zip(axes, DISPARITY.items()):
        ax.bar(x, vals, color=COL6, edgecolor="black", linewidth=0.4)
        ax.set_title(name, fontsize=10.5)
        ax.set_xticks(x); ax.set_xticklabels(METHODS6, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("group accuracy gap\nmajority − minority (↓ = fairer)")
    fig.suptitle("The closed-form reweighter shrinks the majority–minority accuracy gap "
                 "(unfairness) in every regime", fontsize=9.5, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_disparity.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig_disparity.{pdf,png}")


def fig_residual_ablation():
    arms = ["phi1\n(1st-order)", "parallel\n(shrinkage)", "residual\n(real)", "residual\n(shuffle)", "sign_flip"]
    cifar = [0.370, 0.366, 0.458, 0.398, 0.067]
    stl = [0.194, None, 0.323, None, None]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    ax.bar(x - w/2, cifar, w, label="Corrupted-CIFAR (20 seed)", color="#42a5f5", edgecolor="black", linewidth=0.4)
    stl_x = [i for i, v in enumerate(stl) if v is not None]
    ax.bar(np.array(stl_x) + w/2, [stl[i] for i in stl_x], w, label="Spurious-STL10 (10 seed)",
           color="#1565c0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylabel("worst-group accuracy (↑ fairer)")
    ax.set_title("Why it works: the bias-correcting signal is the orthogonal residual\n"
                 "real > shuffle > parallel; sign-flip collapses", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 0.55)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_residual.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig_residual.{pdf,png}")


if __name__ == "__main__":
    fig_three_regimes()
    fig_disparity()
    fig_residual_ablation()
