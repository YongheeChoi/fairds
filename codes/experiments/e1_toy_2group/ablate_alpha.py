"""Quick alpha sweep: does the 2nd-order cross-term sign/scale matter?

We re-use run_one from run.py with majority_ratio=0.9 (the C3-critical setting)
and vary alpha over {-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0}. Three seeds each.

Hypothesis: if our sign convention is wrong, flipping alpha (negative) should
make Δφ (maj - min) decrease as alpha → -|alpha|.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e1_toy_2group.run import run_one  # type: ignore


def main() -> None:
    out_dir = Path("results/e1") / time.strftime("ablate-alpha-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    ratio = 0.9

    runs = []
    for alpha in alphas:
        for seed in seeds:
            tic = time.time()
            rec = run_one(
                method="fairds-2",
                majority_ratio=ratio,
                seed=seed,
                epochs=10,
                n_total=1000,
                n_val=300,
                batch_size=64,
                lr=0.1,
                alpha=alpha,
                device="cuda",
            )
            rec["walltime_sec"] = time.time() - tic
            runs.append(rec)
            phi_maj = rec["phi_per_group_last"].get(0, {}).get("mean", float("nan"))
            phi_min = rec["phi_per_group_last"].get(1, {}).get("mean", float("nan"))
            print(
                f"alpha={alpha:+.2f} seed={seed} acc={rec['final_val_acc']:.3f} "
                f"phi_maj={phi_maj:+.5f} phi_min={phi_min:+.5f} Δφ={phi_maj - phi_min:+.5f}"
            )

    out = {"alphas": alphas, "seeds": seeds, "ratio": ratio, "runs": runs}
    p = out_dir / "ablate_alpha.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")

    # Aggregate
    print("\nAggregate (mean Δφ across seeds):")
    print("alpha  | mean Δφ | mean_phi_maj | mean_phi_min")
    for alpha in alphas:
        vals = [r for r in runs if r["alpha"] == alpha]
        majs = [r["phi_per_group_last"].get(0, {}).get("mean", float("nan")) for r in vals]
        mins = [r["phi_per_group_last"].get(1, {}).get("mean", float("nan")) for r in vals]
        majs = [v for v in majs if not np.isnan(v)]
        mins = [v for v in mins if not np.isnan(v)]
        if majs and mins:
            d = np.mean(majs) - np.mean(mins)
            print(f"{alpha:+.2f}  | {d:+.5f} | {np.mean(majs):+.5f}    | {np.mean(mins):+.5f}")


if __name__ == "__main__":
    main()
