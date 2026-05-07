"""E3 — Colored MNIST sweep (vanilla / fairds-1 / fairds-2 / ren2018).

Standard spurious-correlation benchmark per Codex Round 3 review. Primary
metric is **test-set accuracy** (where spurious is flipped, so models that
relied on color collapse) and **worst-group val accuracy**.

Architecture: small 2-layer CNN (~30k params).
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
from baselines.irm import train_irm  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from datasets.colored_mnist import CMNISTConfig, make_colored_mnist  # noqa: E402


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
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
        out["worst_acc"] = min(out[f"acc_g{0}"], out[f"acc_g{1}"])
    return out


def run_one(*, method: str, seed: int, epochs: int, batch_size: int, lr: float,
            alpha: float, n_train: int, majority_ratio: float, device: str,
            temperature: float = 0.5, weight_scale: float = 1.0) -> dict:
    set_seed(seed)
    cfg = CMNISTConfig(n_train=n_train, majority_ratio=majority_ratio, seed=seed)
    b = make_colored_mnist(cfg)

    Xt = b.X_train.to(device); yt = b.y_train.to(device); gt = b.g_train.to(device)
    # Anchor: USED inside training algorithms (D_val for fairds, meta-loss for ren2018)
    Xa = b.X_anchor.to(device); ya = b.y_anchor.to(device); ga = b.g_anchor.to(device)
    # Held-out eval: NEVER touched during training
    Xv = b.X_val_eval.to(device); yv = b.y_val_eval.to(device); gv = b.g_val_eval.to(device)
    Xte = b.X_test.to(device); yte = b.y_test.to(device); gte = b.g_test.to(device)

    ds = IndexedDS(Xt, yt, gt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = SmallCNN()

    # CRITICAL: algorithms use Xa (anchor), evaluation uses Xv (held-out val_eval).
    # Codex Round 5 fix.
    if method == "vanilla":
        log = train_vanilla(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device,
                            train_x=Xt, train_y=yt)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, Xa, ya, n_train=len(ds),
                           train_groups=gt, order=order, alpha=alpha,
                           temperature=temperature, weight_scale=weight_scale,
                           epochs=epochs, lr=lr, device=device,
                           train_x=Xt, train_y=yt)
    elif method == "ren2018":
        log = train_ren2018(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device,
                            train_x=Xt, train_y=yt)
    elif method == "jtt":
        log = train_jtt(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device,
                        train_x=Xt, train_y=yt)
    elif method == "groupdro":
        log = train_groupdro(model, loader, Xa, ya, n_groups=2,
                             epochs=epochs, lr=lr, device=device,
                             train_x=Xt, train_y=yt)
    elif method == "irm":
        log = train_irm(model, loader, Xa, ya, n_groups=2,
                        epochs=epochs, lr=lr, device=device,
                        train_x=Xt, train_y=yt)
    else:
        raise ValueError(method)

    val_metrics = evaluate(model, Xv, yv, gv, device)  # held-out val_eval
    test_metrics = evaluate(model, Xte, yte, gte, device)

    rec = {
        "method": method, "seed": seed, "majority_ratio": majority_ratio,
        "epochs": epochs, "lr": lr, "alpha": alpha,
        "temperature": temperature, "weight_scale": weight_scale,
        "n_train": n_train,
        # Held-out val_eval metrics (CODEX Round 5 fix — independent of training anchor)
        "val_eval_acc": val_metrics["acc"],
        "val_eval_worst": val_metrics.get("worst_acc"),
        "val_eval_acc_g0": val_metrics.get("acc_g0"),
        "val_eval_acc_g1": val_metrics.get("acc_g1"),
        # OOD test metrics. test groups are: g=0 spurious-aligned, g=1 spurious-flipped.
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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.9])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--weight-scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e3")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e3] writing to {out_dir}")

    runs = []
    total = len(args.methods) * len(args.ratios) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for method in args.methods:
        for ratio in args.ratios:
            for seed in args.seeds:
                counter += 1
                tic = time.time()
                rec = run_one(
                    method=method, seed=seed, majority_ratio=ratio,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, alpha=args.alpha, n_train=args.n_train,
                    temperature=args.temperature, weight_scale=args.weight_scale,
                    device=args.device,
                )
                rec["walltime_sec"] = time.time() - tic
                runs.append(rec)
                print(f"[{counter:>3}/{total}] {method:<10} ratio={ratio:.2f} seed={seed} "
                      f"val_acc={rec['val_eval_acc']:.3f} val_worst={rec.get('val_eval_worst', 0.0):.3f} "
                      f"test_acc={rec['test_acc']:.3f} test_worst={rec.get('test_worst', 0.0):.3f} "
                      f"({rec['walltime_sec']:.1f}s)")

    out = out_dir / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0}, f, indent=2)
    print(f"[e3] {out}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
