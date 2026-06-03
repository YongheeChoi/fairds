"""E6 — Spurious-STL10 sweep (vanilla / fairds-1 / fairds-2 / ren2018 / dfr / groupdro).

The THIRD spurious regime: texture-corruption spurious cue on 96px natural STL-10
photographs (car vs truck). Same anchor / held-out val_eval / OOD test (spurious
flipped) protocol as Colored-MNIST (E3) and Corrupted-CIFAR (E5). Headline metric =
OOD worst-group (spurious-flipped) test accuracy under HONEST val_eval_worst model
selection.

Architecture: small 3-conv CNN trained from scratch (96 -> 12 spatial).
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

from fairds.trainer import train_fairds, train_vanilla  # noqa: E402
from baselines.ren2018 import train_ren2018  # noqa: E402
from baselines.jtt import train_jtt  # noqa: E402
from baselines.groupdro import train_groupdro  # noqa: E402
from baselines.dfr import train_dfr  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from datasets.spurious_stl10 import SpuriousSTL10Config, make_spurious_stl10  # noqa: E402


class StlCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 96 -> 48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 24 -> 12
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, 2),
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


def evaluate(model, X, y, g, device, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            preds.append(model(X[i:i + batch_size].to(device)).argmax(-1).cpu())
    pred = torch.cat(preds)
    y = y.cpu(); g = g.cpu()
    out = {"acc": float((pred == y).float().mean().item())}
    for gid in g.unique().tolist():
        m = (g == gid)
        if m.any():
            out[f"acc_g{gid}"] = float((pred[m] == y[m]).float().mean().item())
    if all(f"acc_g{gid}" in out for gid in (0, 1)):
        out["worst_acc"] = min(out["acc_g0"], out["acc_g1"])
    return out


def run_one(*, method: str, seed: int, epochs: int, batch_size: int, lr: float,
            alpha: float, n_train: int, majority_ratio: float, device: str,
            temperature: float = 0.1, weight_scale: float = 4.0,
            p_corr_majority: float = 0.9, arm: str = None) -> dict:
    set_seed(seed)
    cfg = SpuriousSTL10Config(n_train=n_train, majority_ratio=majority_ratio,
                              p_corr_majority=p_corr_majority, seed=seed)
    b = make_spurious_stl10(cfg)

    Xt = b.X_train.to(device); yt = b.y_train.to(device); gt = b.g_train.to(device)
    Xa = b.X_anchor.to(device); ya = b.y_anchor.to(device)            # used inside algorithms
    Xv = b.X_val_eval; yv = b.y_val_eval; gv = b.g_val_eval           # held-out eval (cpu, eval-only)
    Xte = b.X_test; yte = b.y_test; gte = b.g_test                    # OOD test (cpu, eval-only)

    ds = IndexedDS(Xt, yt, gt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = StlCNN()

    if method == "vanilla":
        log = train_vanilla(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, Xa, ya, n_train=len(ds),
                           train_groups=gt, order=order, alpha=alpha,
                           temperature=temperature, weight_scale=weight_scale,
                           epochs=epochs, lr=lr, device=device, arm=arm)
    elif method == "ren2018":
        log = train_ren2018(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    elif method == "jtt":
        log = train_jtt(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    elif method == "groupdro":
        log = train_groupdro(model, loader, Xa, ya, n_groups=2, epochs=epochs, lr=lr, device=device)
    elif method == "dfr":
        log = train_dfr(model, loader, Xa, ya, train_groups=gt, n_groups=2,
                        epochs=epochs, lr=lr, device=device)
    else:
        raise ValueError(method)

    val_metrics = evaluate(model, Xv, yv, gv, device)
    test_metrics = evaluate(model, Xte, yte, gte, device)

    rec = {
        "method": method, "seed": seed, "majority_ratio": majority_ratio,
        "p_corr_majority": p_corr_majority, "arm": arm,
        "epochs": epochs, "lr": lr, "alpha": alpha,
        "temperature": temperature, "weight_scale": weight_scale, "n_train": n_train,
        "val_eval_acc": val_metrics["acc"],
        "val_eval_worst": val_metrics.get("worst_acc"),
        "val_eval_acc_g0": val_metrics.get("acc_g0"),
        "val_eval_acc_g1": val_metrics.get("acc_g1"),
        "test_acc": test_metrics["acc"],
        "test_worst": test_metrics.get("worst_acc"),
        "test_acc_aligned": test_metrics.get("acc_g0"),
        "test_acc_flipped": test_metrics.get("acc_g1"),
    }
    if hasattr(log, "weight_std_per_batch") and log.weight_std_per_batch:
        rec["w_std_mean"] = float(np.mean(log.weight_std_per_batch))
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["vanilla", "fairds-1", "fairds-2", "ren2018"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.9])
    p.add_argument("--strengths", type=float, nargs="+", default=[0.9])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-train", type=int, default=1400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--weight-scale", type=float, default=4.0)
    p.add_argument("--arm", type=str, default=None,
                   help="residual ablation arm: phi1|parallel|residual_real|residual_shuffle|sign_flip")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e6")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e6] writing to {out_dir}")

    runs = []
    total = len(args.methods) * len(args.ratios) * len(args.strengths) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for method in args.methods:
        for ratio in args.ratios:
            for strength in args.strengths:
                for seed in args.seeds:
                    counter += 1
                    tic = time.time()
                    try:
                        rec = run_one(
                            method=method, seed=seed, majority_ratio=ratio,
                            p_corr_majority=strength,
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, alpha=args.alpha, n_train=args.n_train,
                            temperature=args.temperature, weight_scale=args.weight_scale,
                            device=args.device, arm=args.arm,
                        )
                        rec["walltime_sec"] = time.time() - tic
                        runs.append(rec)
                        print(f"[{counter:>3}/{total}] {method:<12} s={strength:.2f} seed={seed} "
                              f"val_worst={rec.get('val_eval_worst', 0.0):.3f} "
                              f"test_acc={rec['test_acc']:.3f} test_worst={rec.get('test_worst', 0.0):.3f} "
                              f"({rec['walltime_sec']:.1f}s)")
                    except Exception as e:
                        print(f"[{counter}/{total}] {method} s={strength} seed={seed} FAILED: {e}")
                        import traceback; traceback.print_exc()

    out = out_dir / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0}, f, indent=2)
    print(f"[e6] {out}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
