"""Pareto trade-off sweep: Fairds-2 (τ, weight_scale) × Ren2018 (lr) on Adult.

Codex Round 1 review identified the missing piece: we never measured the
accuracy/fairness trade-off curves. Here we sweep settings of each method
that are known to control the strength of reweighting:
  - Fairds-2: temperature τ ∈ {0.1, 0.3, 0.5, 1.0} × weight_scale ∈ {1, 2, 4}
  - Ren2018:   lr ∈ {0.005, 0.01, 0.02, 0.05}  (its weighting magnitude
               scales with the meta-grad step size).
  - Vanilla:   single lr=0.01.

Produces a single CSV with all (acc, dp_diff, eo_diff) so we can plot or
table the Pareto front: does Fairds-2 dominate Ren2018 anywhere?
"""

from __future__ import annotations

import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_adult_compas.run import run_one  # noqa: E402


def main():
    out_dir = Path("results/e2") / time.strftime("pareto-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pareto] writing to {out_dir}")

    seeds = [0, 1, 2]
    runs = []
    t0 = time.time()
    counter = 0

    # Fairds-2 grid: τ × weight_scale
    fairds2_grid = list(product(
        [0.1, 0.3, 0.5, 1.0],   # temperature
        [1.0, 2.0, 4.0],        # weight_scale
    ))

    # Ren2018 grid: meta lr
    ren_grid = [0.005, 0.01, 0.02, 0.05]

    # Vanilla baseline (one config)
    print("[vanilla baseline]")
    for seed in seeds:
        counter += 1
        rec = run_one(
            dataset="adult", method="vanilla", seed=seed, val_mode="balanced",
            epochs=15, batch_size=256, lr=0.01, alpha=0.0,
            n_val_per_group=200, device="cuda",
        )
        rec["config_label"] = "vanilla"
        runs.append(rec)
        print(f"[{counter}] vanilla seed={seed} acc={rec['val_acc']:.3f} dp={rec['dp_diff']:.3f} eo={rec['eo_diff']:.3f}")

    print("\n[fairds-2 grid]")
    for tau, wscale in fairds2_grid:
        for seed in seeds:
            counter += 1
            tic = time.time()
            rec = run_one(
                dataset="adult", method="fairds-2", seed=seed, val_mode="balanced",
                epochs=15, batch_size=256, lr=0.01, alpha=0.5,
                n_val_per_group=200, device="cuda",
                temperature=tau, weight_scale=wscale,
            )
            rec["config_label"] = f"fairds-2/τ{tau}/ws{wscale}"
            rec["walltime_sec"] = time.time() - tic
            runs.append(rec)
            print(f"[{counter}] fairds-2 τ={tau} ws={wscale} seed={seed} "
                  f"acc={rec['val_acc']:.3f} dp={rec['dp_diff']:.3f} eo={rec['eo_diff']:.3f} "
                  f"w_std={rec.get('w_std_mean','—')}")

    print("\n[ren2018 grid]")
    for lr in ren_grid:
        for seed in seeds:
            counter += 1
            tic = time.time()
            rec = run_one(
                dataset="adult", method="ren2018", seed=seed, val_mode="balanced",
                epochs=15, batch_size=256, lr=lr, alpha=0.0,
                n_val_per_group=200, device="cuda",
            )
            rec["config_label"] = f"ren2018/lr{lr}"
            rec["walltime_sec"] = time.time() - tic
            runs.append(rec)
            print(f"[{counter}] ren2018 lr={lr} seed={seed} "
                  f"acc={rec['val_acc']:.3f} dp={rec['dp_diff']:.3f} eo={rec['eo_diff']:.3f}")

    # Save JSON + CSV
    out_json = out_dir / "pareto.json"
    out_json.write_text(json.dumps({"runs": runs, "total_walltime_sec": time.time() - t0}, indent=2))
    out_csv = out_dir / "pareto.csv"
    with open(out_csv, "w", newline="") as f:
        keys = ["config_label", "method", "seed", "val_acc", "worst_acc",
                "dp_diff", "eo_diff", "temperature", "weight_scale", "lr",
                "w_std_mean", "w_max_mean", "w_min_mean", "phi_std_mean"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"\n[pareto] {out_json}, {out_csv}")
    print(f"[pareto] total walltime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
