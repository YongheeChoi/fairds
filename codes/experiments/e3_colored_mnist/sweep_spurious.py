"""Spurious-strength sweep: how does Fairds-2's gain depend on the
strength of the spurious correlation in Colored MNIST?

Vary `p_color_majority` ∈ {0.7, 0.8, 0.9, 0.95, 0.99} (the probability
that color matches the label in the majority training group). All other
hyperparameters fixed at the best-tuned setting from §4.1.

Reports test_worst (mean ± std over 5 seeds) per method per spurious
strength. Generates a single figure showing the regime where the
2nd-order isolation is most prominent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairds.trainer import train_fairds, train_vanilla
from baselines.ren2018 import train_ren2018
from utils.seed import set_seed
from datasets.colored_mnist import CMNISTConfig, make_colored_mnist


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class IndexedDS(torch.utils.data.Dataset):
    def __init__(self, x, y, g):
        self.x, self.y, self.g = x, y, g

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i, self.g[i]


def evaluate(model, X, y, g, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device); y = y.to(device); g = g.to(device)
        pred = model(X).argmax(-1)
    out = {"acc": float((pred == y).float().mean().item())}
    for gid in g.unique().cpu().tolist():
        m = (g == gid)
        if m.any():
            out[f"acc_g{gid}"] = float((pred[m] == y[m]).float().mean().item())
    if all(f"acc_g{gid}" in out for gid in (0, 1)):
        out["worst_acc"] = min(out[f"acc_g0"], out[f"acc_g1"])
    return out


def run_one(method, seed, p_spurious, epochs, lr, alpha, temperature, weight_scale, device):
    set_seed(seed)
    cfg = CMNISTConfig(
        n_train=5000, p_color_majority=p_spurious, p_color_minority=0.5,
        p_color_test=1.0 - p_spurious,  # flipped at test
        majority_ratio=0.9, seed=seed,
    )
    b = make_colored_mnist(cfg)
    Xt = b.X_train.to(device); yt = b.y_train.to(device); gt = b.g_train.to(device)
    Xa = b.X_anchor.to(device); ya = b.y_anchor.to(device)
    Xv = b.X_val_eval.to(device); yv = b.y_val_eval.to(device); gv = b.g_val_eval.to(device)
    Xte = b.X_test.to(device); yte = b.y_test.to(device); gte = b.g_test.to(device)

    ds = IndexedDS(Xt, yt, gt)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = SmallCNN()
    if method == "vanilla":
        log = train_vanilla(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device, train_x=Xt, train_y=yt)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, Xa, ya, n_train=len(ds),
                           train_groups=gt, order=order, alpha=alpha,
                           temperature=temperature, weight_scale=weight_scale,
                           epochs=epochs, lr=lr, device=device, train_x=Xt, train_y=yt)
    elif method == "ren2018":
        log = train_ren2018(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device, train_x=Xt, train_y=yt)

    val = evaluate(model, Xv, yv, gv, device)
    test = evaluate(model, Xte, yte, gte, device)
    return {
        "method": method, "seed": seed, "p_spurious": p_spurious,
        "val_eval_worst": val["worst_acc"], "test_acc": test["acc"], "test_worst": test["worst_acc"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strengths", type=float, nargs="+", default=[0.7, 0.8, 0.9, 0.95, 0.99])
    p.add_argument("--methods", nargs="+", default=["vanilla", "fairds-1", "fairds-2", "ren2018"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--weight-scale", type=float, default=4.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e3_strength")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[strength] writing to {out_dir}")

    runs = []
    total = len(args.methods) * len(args.strengths) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for method in args.methods:
        for s in args.strengths:
            for seed in args.seeds:
                counter += 1
                rec = run_one(method, seed, s, args.epochs, args.lr, args.alpha,
                              args.temperature, args.weight_scale, args.device)
                runs.append(rec)
                print(f"[{counter:>3}/{total}] {method:<10} p={s:.2f} seed={seed} "
                      f"test_worst={rec['test_worst']:.3f}")
    out = out_dir / "sweep_results.json"
    out.write_text(json.dumps({"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0}, indent=2))
    print(f"[strength] {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
