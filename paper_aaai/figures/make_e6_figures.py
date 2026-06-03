"""Figures for the 3-regime Fairds paper (Colored MNIST / Corrupted-CIFAR / Spurious-STL10).

Saves both PDF (for LaTeX) and PNG (for the weasyprint HTML slides).
Numbers are the confirmed 10-seed test_worst values from findings.md / analyze.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11, "axes.spysize": 11} if False else {"font.size": 11})

# --- Confirmed test_worst (mean) across three from-scratch spurious regimes ---
METHODS = ["vanilla", "fairds-1", "fairds-2", "fairds-2\nresidual", "ren2018", "jtt", "groupdro"]
SHORT = ["vanilla", "fairds-1", "fairds-2", "fairds-2-resid", "ren2018", "jtt", "groupdro*"]
REGIMES = {
    "Colored MNIST\n(color)":      [0.694, 0.777, 0.824, 0.815, 0.796, 0.878, 0.900],
    "Corrupted-CIFAR\n(texture)":  [0.196, 0.374, 0.423, 0.458, 0.375, 0.368, 0.564],
    "Spurious-STL10\n(natural img)": [0.042, 0.194, 0.227, 0.323, 0.151, 0.606, 0.521],
}
# std where available (STL10 from analyze.py); used only for the STL panel error bars
STL_STD = [0.026, 0.138, 0.154, 0.111, 0.124, 0.043, 0.043]

# color: fairds family = blues (resid darkest), baselines = grey, oracle/2-stage = red/orange
COLORS = ["#9e9e9e", "#90caf9", "#42a5f5", "#1565c0", "#bdbdbd", "#ef9a9a", "#c62828"]


def fig_three_regimes():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    x = np.arange(len(METHODS))
    for ax, (name, vals) in zip(axes, REGIMES.items()):
        yerr = STL_STD if "STL" in name else None
        bars = ax.bar(x, vals, color=COLORS, yerr=yerr, capsize=2, edgecolor="black", linewidth=0.4)
        ax.set_title(name, fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        # highlight the residual bar
        bars[3].set_hatch("///")
    axes[0].set_ylabel("OOD worst-group test acc")
    fig.suptitle("Fairds-2 (residual) beats no-group-label baselines across 3 spurious regimes  "
                 "(grey/blue = no group label; red = uses group label or 2-stage)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_three_regimes.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig_three_regimes.{pdf,png}")


def fig_residual_ablation():
    # Residual decomposition arms. CIFAR = 20 seeds, STL = 10 seeds (per analyze.py / findings).
    arms = ["phi1\n(1st-order)", "parallel\n(shrinkage)", "residual\n(real)", "residual\n(shuffle)", "sign_flip"]
    cifar = [0.370, 0.366, 0.458, 0.398, 0.067]
    stl =   [0.194, None,  0.323, None,  None]   # STL: only phi1 & residual_real measured at 10 seed
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    ax.bar(x - w/2, cifar, w, label="Corrupted-CIFAR (20 seed)", color="#42a5f5", edgecolor="black", linewidth=0.4)
    stl_x = [i for i, v in enumerate(stl) if v is not None]
    stl_v = [stl[i] for i in stl_x]
    ax.bar(np.array(stl_x) + w/2, stl_v, w, label="Spurious-STL10 (10 seed)", color="#1565c0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylabel("OOD worst-group test acc")
    ax.set_title("Cross-term decomposition: the orthogonal residual carries the 2nd-order signal\n"
                 "real > shuffle > parallel; sign-flip collapses", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.55)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_residual.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig_residual.{pdf,png}")


if __name__ == "__main__":
    fig_three_regimes()
    fig_residual_ablation()
