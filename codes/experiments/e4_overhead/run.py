"""E4 — Wall-clock overhead measurement on CIFAR-10 + ResNet-18.

Tests claim C1 at a realistic-scale model. plan.md target: 1st-order ≤1.5×,
2nd-order ≤3.0× of vanilla. With our vmap implementation we measured
5–7× on small CNN/MLP. This experiment measures whether the ratio
shrinks at ResNet-18 scale (where vanilla forward/backward is heavier).

Setup:
- CIFAR-10, 10k subset (for time), batch 256
- ResNet-18 from scratch (BN frozen for vmap)
- 3 epochs, time per step (excluding eval)
- 3 seeds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairds.trainer import train_fairds, train_vanilla  # noqa: E402
from baselines.ren2018 import train_ren2018  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def _build_model(num_classes=10):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.eval()
            mod.track_running_stats = False
    return m


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i, 0  # dummy group


def get_data(n_train: int, n_anchor: int, batch_size: int, device: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    full = datasets.CIFAR10(root="~/.cache/fairds_data/cifar10", train=True, download=True, transform=transform)
    indices = list(range(n_train + n_anchor))
    train_idx = indices[:n_train]
    anchor_idx = indices[n_train:]
    train_ds = IndexedSubset(Subset(full, train_idx))
    anchor_ds = Subset(full, anchor_idx)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    Xa = torch.stack([anchor_ds[i][0] for i in range(len(anchor_ds))]).to(device)
    ya = torch.tensor([anchor_ds[i][1] for i in range(len(anchor_ds))], dtype=torch.long).to(device)
    return loader, Xa, ya, len(train_ds)


def run_one(method: str, seed: int, epochs: int, batch_size: int, lr: float,
            n_train: int, n_anchor: int, device: str) -> dict:
    set_seed(seed)
    loader, Xa, ya, n = get_data(n_train, n_anchor, batch_size, device)
    train_groups = torch.zeros(n, dtype=torch.long, device=device)
    model = _build_model().to(device)

    # Warm GPU
    if device == "cuda":
        torch.cuda.synchronize()

    tic = time.time()
    if method == "vanilla":
        log = train_vanilla(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, Xa, ya, n_train=n,
                           train_groups=train_groups, order=order, alpha=0.5,
                           temperature=0.5, weight_scale=1.0,
                           epochs=epochs, lr=lr, device=device)
    elif method == "ren2018":
        log = train_ren2018(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    else:
        raise ValueError(method)
    if device == "cuda":
        torch.cuda.synchronize()
    walltime = time.time() - tic

    return {
        "method": method, "seed": seed, "epochs": epochs, "lr": lr,
        "n_train": n_train, "batch_size": batch_size,
        "wall_sec": walltime,
        "wall_per_epoch": walltime / epochs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["vanilla", "fairds-1", "fairds-2", "ren2018"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-anchor", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e4")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e4] writing to {out_dir}")

    runs = []
    total = len(args.methods) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for method in args.methods:
        for seed in args.seeds:
            counter += 1
            try:
                rec = run_one(method, seed, args.epochs, args.batch_size, args.lr,
                              args.n_train, args.n_anchor, args.device)
                runs.append(rec)
                print(f"[{counter:>2}/{total}] {method:<10} seed={seed} "
                      f"wall={rec['wall_sec']:.1f}s ({rec['wall_per_epoch']:.1f}s/ep)")
            except Exception as e:
                print(f"[{counter}/{total}] {method} seed={seed} FAILED: {e}")
                import traceback; traceback.print_exc()

    out = out_dir / "sweep_results.json"
    out.write_text(json.dumps({"args": vars(args), "runs": runs, "total_walltime_sec": time.time()-t0}, indent=2))
    print(f"[e4] {out}")

    # quick aggregate
    import numpy as np
    from collections import defaultdict
    agg = defaultdict(list)
    for r in runs:
        agg[r["method"]].append(r["wall_per_epoch"])
    if "vanilla" in agg:
        v = np.mean(agg["vanilla"])
        print("\nOverhead ratio (mean per-epoch wall, vs vanilla):")
        for m in args.methods:
            if m in agg:
                ratio = np.mean(agg[m]) / v
                print(f"  {m}: {np.mean(agg[m]):.1f}s/ep ({ratio:.2f}× vanilla)")


if __name__ == "__main__":
    main()
