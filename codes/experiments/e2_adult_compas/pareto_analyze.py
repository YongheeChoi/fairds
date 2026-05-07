"""Aggregate the pareto sweep into method-level (acc, dp_diff, eo_diff)
mean-over-seeds points and identify the Pareto front.

Question: at any acc level, does fairds-2 dominate ren2018 on (dp_diff,
eo_diff)? Or vice versa?
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_path", type=str)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    with open(a.results_path) as f:
        runs = json.load(f)["runs"]

    agg = defaultdict(lambda: defaultdict(list))
    for r in runs:
        cfg = r["config_label"]
        agg[cfg]["acc"].append(r["val_acc"])
        agg[cfg]["worst"].append(r["worst_acc"])
        agg[cfg]["dp"].append(r["dp_diff"])
        agg[cfg]["eo"].append(r["eo_diff"])
        agg[cfg]["method"] = r["method"]
        if "w_std_mean" in r:
            agg[cfg]["w_std"].append(r["w_std_mean"])

    points = []
    for cfg, d in agg.items():
        points.append({
            "config": cfg,
            "method": d.get("method", ""),
            "acc": float(np.mean(d["acc"])),
            "acc_std": float(np.std(d["acc"])),
            "worst": float(np.mean(d["worst"])),
            "dp": float(np.mean(d["dp"])),
            "dp_std": float(np.std(d["dp"])),
            "eo": float(np.mean(d["eo"])),
            "eo_std": float(np.std(d["eo"])),
            "w_std": float(np.mean(d.get("w_std", [0]))) if "w_std" in d else None,
        })

    # Sort by accuracy
    points.sort(key=lambda x: -x["acc"])

    L = []
    L.append("# E2 Adult — Pareto trade-off (fairds-2 grid vs ren2018 grid vs vanilla)")
    L.append("")
    L.append("Mean over 3 seeds. **Bold** = Pareto-optimal in (acc, dp+eo) space (lower dp/eo better, higher acc better).")
    L.append("")
    L.append("| config | method | val_acc | worst | dp_diff | eo_diff | dp+eo | w_std |")
    L.append("|---|---|---|---|---|---|---|---|")

    # Mark Pareto-optimal (in (acc, dp+eo) where we want high acc AND low dp+eo)
    # A point is dominated if there exists another point with acc >= this AND dp+eo <= this (with at least one strict)
    def dominated(i, pts):
        for j, q in enumerate(pts):
            if i == j:
                continue
            cond = (q["acc"] >= pts[i]["acc"] and (q["dp"] + q["eo"]) <= (pts[i]["dp"] + pts[i]["eo"]))
            strict = (q["acc"] > pts[i]["acc"] or (q["dp"] + q["eo"]) < (pts[i]["dp"] + pts[i]["eo"]))
            if cond and strict:
                return True
        return False

    for i, pt in enumerate(points):
        is_pareto = not dominated(i, points)
        cfg = f"**{pt['config']}**" if is_pareto else pt["config"]
        w_std_str = "—" if pt["w_std"] is None else f"{pt['w_std']:.3f}"
        L.append(
            f"| {cfg} | {pt['method']} | {pt['acc']:.3f}±{pt['acc_std']:.3f} "
            f"| {pt['worst']:.3f} | {pt['dp']:.3f}±{pt['dp_std']:.3f} | {pt['eo']:.3f}±{pt['eo_std']:.3f} "
            f"| {(pt['dp']+pt['eo']):.3f} | {w_std_str} |"
        )

    L.append("")
    L.append("## Pareto-optimal configurations (high acc AND low dp+eo)")
    L.append("")
    pareto_pts = [p for i, p in enumerate(points) if not dominated(i, points)]
    pareto_pts.sort(key=lambda p: -p["acc"])
    L.append("| config | method | acc | dp_diff | eo_diff | dp+eo |")
    L.append("|---|---|---|---|---|---|")
    for p in pareto_pts:
        L.append(f"| {p['config']} | {p['method']} | {p['acc']:.3f} | {p['dp']:.3f} | {p['eo']:.3f} | {p['dp']+p['eo']:.3f} |")

    L.append("")
    L.append("## Best per-method (lowest dp+eo subject to acc ≥ vanilla − 1pp)")
    L.append("")
    vanilla_acc = next((p["acc"] for p in points if p["method"] == "vanilla"), None)
    if vanilla_acc:
        threshold = vanilla_acc - 0.01
        L.append(f"vanilla acc = {vanilla_acc:.3f}, threshold = {threshold:.3f}")
        L.append("")
        L.append("| method | best config | acc | dp_diff | eo_diff | dp+eo | Δ vs vanilla(dp+eo) |")
        L.append("|---|---|---|---|---|---|---|")
        van_dpeo = next((p["dp"] + p["eo"] for p in points if p["method"] == "vanilla"), None)
        for method in ("fairds-2", "ren2018"):
            cands = [p for p in points if p["method"] == method and p["acc"] >= threshold]
            if cands:
                best = min(cands, key=lambda p: p["dp"] + p["eo"])
                L.append(f"| {method} | {best['config']} | {best['acc']:.3f} | {best['dp']:.3f} | {best['eo']:.3f} | {best['dp']+best['eo']:.3f} | {best['dp']+best['eo']-van_dpeo:+.3f} |")

    L.append("")
    L.append("## Diagnostic: weight magnitude (fairds-2 only)")
    L.append("")
    L.append("Higher w_std means stronger reweighting. Codex Round 1 noted Δw was tiny — does τ↓ or weight_scale↑ fix it?")
    L.append("")
    L.append("| config | w_std | acc | dp+eo |")
    L.append("|---|---|---|---|")
    f2 = sorted([p for p in points if p["method"] == "fairds-2" and p["w_std"] is not None], key=lambda p: -p["w_std"])
    for p in f2:
        L.append(f"| {p['config']} | {p['w_std']:.3f} | {p['acc']:.3f} | {p['dp']+p['eo']:.3f} |")

    txt = "\n".join(L)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"[pareto-analyze] wrote {a.out}")
    print(txt)


if __name__ == "__main__":
    main()
