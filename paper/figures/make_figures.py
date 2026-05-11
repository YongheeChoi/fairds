"""Generate paper figures + tables from result JSONs.

Outputs:
  paper/figures/fig_cmnist_results.pdf  — Colored MNIST main result bars
  paper/figures/fig_waterbirds.pdf       — Waterbirds 3-variant matrix
  paper/figures/fig_e1b_phi.pdf          — E1b phi-by-group across imbalance
  paper/figures/fig_pareto_adult.pdf     — Adult Pareto trade-off (appendix)
  paper/tables/cmnist_main.tex           — main result LaTeX table
  paper/tables/cmnist_paired.tex         — paired t-test LaTeX table
  paper/tables/waterbirds_grid.tex       — Waterbirds 3-variant table
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent
TBL_DIR = PROJECT_ROOT / "paper" / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality matplotlib style
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "vanilla": "#888888",
    "fairds-1": "#4C72B0",
    "fairds-2": "#0F9D58",   # green = positive in cmnist
    "ren2018": "#DB4437",
}
METHOD_LABEL = {
    "vanilla": "Vanilla",
    "fairds-1": "Fairds-1",
    "fairds-2": "Fairds-2",
    "ren2018": "FORML",
}


def load_runs(rel_path: str):
    p = PROJECT_ROOT / rel_path
    return json.loads(p.read_text())["runs"]


def cmnist_main_table_and_fig():
    """Colored MNIST main result table + bar figure (held-out + OOD)."""
    runs = load_runs("results/e3/20260506-190605/sweep_results.json")
    by_m = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = r["method"]
        by_m[m]["val_eval_worst"].append(r["val_eval_worst"])
        by_m[m]["test_acc"].append(r["test_acc"])
        by_m[m]["test_worst"].append(r["test_worst"])

    methods = ["vanilla", "ren2018", "fairds-1", "fairds-2"]
    metrics = ["val_eval_worst", "test_acc", "test_worst"]
    metric_labels = [r"Held-out worst (val\_eval)", r"OOD test acc", r"OOD test worst"]

    # ---- Table 1: main results ----
    rows = []
    rows.append(r"\begin{tabular}{lccc}")
    rows.append(r"\toprule")
    rows.append(r"Method & " + " & ".join(metric_labels) + r" \\")
    rows.append(r"\midrule")
    for m in methods:
        d = by_m[m]
        cells = []
        for k in metrics:
            v = np.array(d[k])
            cells.append(f"{v.mean():.3f} $\\pm$ {v.std():.3f}")
        is_best = (m == "fairds-2")
        line = METHOD_LABEL[m]
        if is_best:
            line = r"\textbf{" + line + "}"
            cells = [r"\textbf{" + c + "}" for c in cells]
        rows.append(line + " & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (TBL_DIR / "cmnist_main.tex").write_text("\n".join(rows))

    # ---- Table 2: paired t-tests ----
    v_w = np.array(by_m["vanilla"]["val_eval_worst"])
    v_t = np.array(by_m["vanilla"]["test_acc"])
    v_tw = np.array(by_m["vanilla"]["test_worst"])
    rrs = []
    rrs.append(r"\begin{tabular}{lcccc}")
    rrs.append(r"\toprule")
    rrs.append(r"Comparison & $\Delta$ val\_worst & $\Delta$ test\_acc & $\Delta$ test\_worst & $p$ (test\_worst) \\")
    rrs.append(r"\midrule")
    for m in ("fairds-1", "fairds-2", "ren2018"):
        mw = np.array(by_m[m]["val_eval_worst"])
        mt = np.array(by_m[m]["test_acc"])
        mtw = np.array(by_m[m]["test_worst"])
        d_w = (mw - v_w).mean()
        d_t = (mt - v_t).mean()
        d_tw = (mtw - v_tw).mean()
        t, p2 = ttest_rel(mtw, v_tw)
        p = p2 / 2 if t > 0 else 1 - p2 / 2
        rrs.append(f"{METHOD_LABEL[m]} vs Vanilla & {d_w:+.4f} & {d_t:+.4f} & {d_tw:+.4f} & {p:.3g} \\\\")
    # Fairds-2 vs Ren2018
    f2w = np.array(by_m["fairds-2"]["val_eval_worst"])
    f2t = np.array(by_m["fairds-2"]["test_acc"])
    f2tw = np.array(by_m["fairds-2"]["test_worst"])
    rw = np.array(by_m["ren2018"]["val_eval_worst"])
    rt = np.array(by_m["ren2018"]["test_acc"])
    rtw = np.array(by_m["ren2018"]["test_worst"])
    t, p2 = ttest_rel(f2tw, rtw)
    p = p2 / 2 if t > 0 else 1 - p2 / 2
    rrs.append(f"Fairds-2 vs Ren2018 & {(f2w-rw).mean():+.4f} & {(f2t-rt).mean():+.4f} & {(f2tw-rtw).mean():+.4f} & {p:.3g} \\\\")
    # Isolation
    f1w = np.array(by_m["fairds-1"]["val_eval_worst"])
    f1t = np.array(by_m["fairds-1"]["test_acc"])
    f1tw = np.array(by_m["fairds-1"]["test_worst"])
    t, p2 = ttest_rel(f2tw, f1tw)
    p = p2 / 2 if t > 0 else 1 - p2 / 2
    rrs.append(r"Fairds-2 vs Fairds-1 \emph{(2nd-order isolation)} & " +
               f"{(f2w-f1w).mean():+.4f} & {(f2t-f1t).mean():+.4f} & {(f2tw-f1tw).mean():+.4f} & {p:.3g} \\\\")
    rrs.append(r"\bottomrule")
    rrs.append(r"\end{tabular}")
    (TBL_DIR / "cmnist_paired.tex").write_text("\n".join(rrs))

    # ---- Fig: bar plot test_worst ----
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    means = [np.mean(by_m[m]["test_worst"]) for m in methods]
    stds = [np.std(by_m[m]["test_worst"]) for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    xs = np.arange(len(methods))
    bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("OOD worst-group acc")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.694, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(3.5, 0.83, "+13 pp\n(p=5e-5)", fontsize=7, ha="center", color="#0F9D58", fontweight="bold")
    ax.set_title("Colored MNIST: where Fairds-2 wins", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cmnist_results.pdf")
    plt.close(fig)


def waterbirds_table_and_fig():
    """Waterbirds 3-variant matrix table + bar figure."""
    cfgs = [
        ("default-hp", "results/e3b/20260507-002752/sweep_results.json"),
        ("warmup=3",  "results/e3b_warmup3/20260507-003831/sweep_results.json"),
        ("no\\_ema",  "results/e3b_no_ema/20260507-005213/sweep_results.json"),
    ]
    methods = ["vanilla", "fairds-1", "fairds-2", "ren2018"]
    grid = {cfg: defaultdict(list) for cfg, _ in cfgs}
    for cfg, p in cfgs:
        for r in load_runs(p):
            grid[cfg][r["method"]].append(r["test_worst"])

    # ---- Table: 3-variant matrix ----
    rows = []
    rows.append(r"\begin{tabular}{lccc}")
    rows.append(r"\toprule")
    rows.append(r"Method & " + " & ".join(c for c, _ in cfgs) + r" \\")
    rows.append(r"\midrule")
    for m in methods:
        cells = []
        for cfg, _ in cfgs:
            vs = grid[cfg][m]
            cells.append(f"{np.mean(vs):.3f} $\\pm$ {np.std(vs):.3f}")
        rows.append(METHOD_LABEL[m] + " & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (TBL_DIR / "waterbirds_grid.tex").write_text("\n".join(rows))

    # ---- Fig: grouped bars (best variant per fairds = warmup3, ren2018 unchanged) ----
    # Use warmup3 row since it's the strongest fairds variant
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    cfg_use = "warmup=3"
    means = [np.mean(grid[cfg_use][m]) for m in methods]
    stds = [np.std(grid[cfg_use][m]) for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    # Recolor fairds bars red on Waterbirds (failure marker)
    colors = ["#888888", "#C0392B", "#C0392B", "#0F9D58"]
    xs = np.arange(len(methods))
    bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("OOD worst-group acc")
    ax.set_ylim(0, 1.0)
    ax.text(3.0, 0.55, "FORML wins\n(p=0.02 vs vanilla)",
            fontsize=7, ha="center", color="#0F9D58", fontweight="bold")
    ax.text(1.5, 0.32, "Fairds fails\n(3 variants tried)",
            fontsize=7, ha="center", color="#C0392B", fontweight="bold")
    ax.set_title("Waterbirds: where Fairds-2 fails", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_waterbirds.pdf")
    plt.close(fig)


def e1b_phi_fig():
    """E1b — per-group phi statistics across imbalance ratios."""
    runs = load_runs("results/e1b/20260506-153909/sweep_results.json")
    by_method = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if not r["method"].startswith("fairds"):
            continue
        ratio = r["majority_ratio"]
        ppl = r.get("phi_per_group_last", {})
        for gid in ("0", "1"):
            stats = ppl.get(gid, {})
            if "mean" in stats:
                by_method[r["method"]][(ratio, gid)].append(stats["mean"])

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True)
    ratios = [0.5, 0.7, 0.9, 0.99]
    for ax, m in zip(axes, ["fairds-1", "fairds-2"]):
        x = np.arange(len(ratios))
        maj = [np.mean(by_method[m].get((r, "0"), [np.nan])) for r in ratios]
        mn = [np.mean(by_method[m].get((r, "1"), [np.nan])) for r in ratios]
        ax.plot(x, maj, "o-", color="#888888", label="majority")
        ax.plot(x, mn, "s-", color="#0F9D58", label="minority")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.2f}" for r in ratios])
        ax.set_xlabel("majority ratio")
        ax.set_title(METHOD_LABEL[m], fontsize=9)
        ax.axhline(0.0, color="black", linewidth=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel(r"$\phi_i$ (mean per group)")
    fig.suptitle("E1b mechanism: 2nd-order cross-term down-weights majority $\phi_i$ as imbalance grows", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_e1b_phi.pdf")
    plt.close(fig)


def adult_pareto_fig():
    """Appendix: Adult Pareto trade-off."""
    runs = load_runs("results/e2/pareto-20260506-181137/pareto.json")
    pts = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = r["method"]
        pts[m]["acc"].append(r["val_acc"])
        pts[m]["dpeo"].append(r["dp_diff"] + r["eo_diff"])

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in ("vanilla", "fairds-2", "ren2018"):
        if m not in pts: continue
        ax.scatter(pts[m]["dpeo"], pts[m]["acc"],
                   color=METHOD_COLORS[m], label=METHOD_LABEL[m], alpha=0.7, s=24)
    ax.set_xlabel("DP-diff + EO-diff (lower better)")
    ax.set_ylabel("val acc (higher better)")
    ax.set_title("Adult: Pareto trade-off (each point = one config $\\times$ seed)", fontsize=9)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pareto_adult.pdf")
    plt.close(fig)


def main():
    print("[fig] Colored MNIST main result + tables")
    cmnist_main_table_and_fig()
    print("[fig] Waterbirds 3-variant + table")
    waterbirds_table_and_fig()
    print("[fig] E1b phi mechanism")
    e1b_phi_fig()
    print("[fig] Adult Pareto (appendix)")
    adult_pareto_fig()
    print("[fig] Done.")


if __name__ == "__main__":
    main()
