"""Quick alpha + anchor-size ablation on Adult.

E2 main sweep showed weak fairness improvement vs vanilla. Two hypotheses:
  H1 (alpha): cross-term magnitude with the default RMS rescaling and
      alpha=0.5 leaves weights nearly uniform (Δw ~ 0.01). A larger alpha
      may produce stronger reweighting.
  H2 (anchor): n_per_group=200 (n_val=400) is too small for the Adult
      val set (~9k); a larger anchor stabilizes the gradient direction
      and lets fairds make more confident reweighting decisions.

We do NOT touch lr/epochs to keep the comparison apples-to-apples with
the main sweep.
"""

from __future__ import annotations

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
    out_dir = Path("results/e2") / time.strftime("ablate-%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    alphas = [0.5, 1.0, 2.0, 4.0]
    anchor_sizes = [200, 500, 1000]

    runs = []
    for alpha, n_anchor, seed in product(alphas, anchor_sizes, seeds):
        tic = time.time()
        rec = run_one(
            dataset="adult", method="fairds-2", seed=seed, val_mode="balanced",
            epochs=15, batch_size=256, lr=0.01, alpha=alpha,
            n_val_per_group=n_anchor, device="cuda",
        )
        rec["walltime_sec"] = time.time() - tic
        runs.append(rec)
        print(f"alpha={alpha:>4.1f} anchor={n_anchor:>4} seed={seed} "
              f"acc={rec['val_acc']:.3f} dp={rec['dp_diff']:.3f} eo={rec['eo_diff']:.3f}")

    out = out_dir / "ablate.json"
    out.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\nwrote {out}")

    # quick aggregation
    import numpy as np
    from collections import defaultdict
    agg = defaultdict(list)
    for r in runs:
        agg[(r["alpha"], r["n_val_per_group"])].append(r)
    print("\nAggregate over 3 seeds (Adult, fairds-2):")
    print(f"{'alpha':<6} {'anchor':<6} {'val_acc':<10} {'dp_diff':<10} {'eo_diff':<10}")
    for (a, k), rs in sorted(agg.items()):
        accs = [x["val_acc"] for x in rs]
        dps = [x["dp_diff"] for x in rs]
        eos = [x["eo_diff"] for x in rs]
        print(f"{a:<6.1f} {k:<6} {np.mean(accs):.3f}±{np.std(accs):.3f} "
              f"{np.mean(dps):.3f}±{np.std(dps):.3f} {np.mean(eos):.3f}±{np.std(eos):.3f}")


if __name__ == "__main__":
    main()
