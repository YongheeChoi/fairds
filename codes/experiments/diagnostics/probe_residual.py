"""Why is the residual signal benchmark-dependent?
   Corrupted-CIFAR (texture spurious): residual_real vs shuffle p=0.0015 (strong)
   Colored MNIST   (color spurious):   residual_real vs shuffle p=0.26   (weak)

Hypothesis: under color spurious the cross-term is almost perfectly parallel to
the first-order Shapley score (tiny orthogonal residual), whereas texture spurious
yields a larger, more group-discriminative residual. We measure per from-scratch
regime:
  - cos(phi1, cross_n)        -> residual_fraction = sqrt(1 - cos^2)
  - r group-separation (minority - majority), RMS-matched to phi1's, so the ratio
    r_sep / phi1_sep says how much group signal lives in the residual vs first-order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairds.shapley import shapley_residual_arms  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from datasets.colored_mnist import CMNISTConfig, make_colored_mnist  # noqa: E402
from datasets.corrupted_cifar import CorruptedCIFARConfig, make_corrupted_cifar  # noqa: E402
from experiments.e3_colored_mnist.run import SmallCNN  # noqa: E402
from experiments.e5_corrupted_cifar.run import CifarCNN  # noqa: E402
from experiments.diagnostics.probe import quick_train  # noqa: E402


def probe_residual(model, xt, yt, gt, xv, yv, n_sub=400):
    idx = torch.randperm(len(xt))[:n_sub]
    phi1, cn, r, beta = shapley_residual_arms(model, F.cross_entropy, xt[idx], yt[idx], xv, yv)
    cos = float((phi1 @ cn) / (phi1.norm() * cn.norm()).clamp_min(1e-12))
    resid_frac = float(max(0.0, 1.0 - cos * cos) ** 0.5)
    g = gt[idx]
    maj, mn = (g == 0), (g == 1)
    r_sep = float(r[mn].mean() - r[maj].mean())          # r is RMS-matched to phi1
    phi1_sep = float(phi1[mn].mean() - phi1[maj].mean())
    return {
        "cos_first_cross": round(cos, 4),
        "residual_fraction": round(resid_frac, 4),
        "r_group_sep": round(r_sep, 3),
        "phi1_group_sep": round(phi1_sep, 3),
        "r_sep_over_phi1_sep": round(r_sep / (phi1_sep if abs(phi1_sep) > 1e-9 else 1e-9), 3),
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    reps = 10
    print("Residual structure by spurious modality (mean over 10 seeds)\n")
    for name, kind in [("CMNIST (color)", "cmnist"), ("Corrupted-CIFAR (texture)", "cifar")]:
        accum = []
        for seed in range(reps):
            set_seed(seed)
            if kind == "cmnist":
                b = make_colored_mnist(CMNISTConfig(seed=seed))
                model = SmallCNN().to(dev); lr = 0.05
            else:
                b = make_corrupted_cifar(CorruptedCIFARConfig(seed=seed))
                model = CifarCNN().to(dev); lr = 0.02
            xt, yt, gt = b.X_train.to(dev), b.y_train.to(dev), b.g_train.to(dev)
            xa, ya = b.X_anchor.to(dev), b.y_anchor.to(dev)
            quick_train(model, xt, yt, gt, xa, ya, dev, epochs=8, lr=lr)
            accum.append(probe_residual(model, xt, yt, gt, xa, ya))
        avg = {k: round(float(np.mean([a[k] for a in accum])), 4) for k in accum[0]}
        aligns = [a["r_sep_over_phi1_sep"] for a in accum]
        npos = sum(1 for x in aligns if x > 0)
        print(f"  {name:26s}: residual_fraction={avg['residual_fraction']:.3f}  "
              f"r_sep/phi1_sep={avg['r_sep_over_phi1_sep']:+.2f}  "
              f"(aligned-positive in {npos}/{reps} seeds)")


if __name__ == "__main__":
    main()
