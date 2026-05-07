"""E1b — Spurious-feature 2-group MLP sweep.

This is the *real* C3 testbed: a 2-layer MLP with a clear spurious feature
that the majority group can exploit but the minority cannot. If the 2nd-order
cross-term truly attenuates representation bias (the plan's G5 hypothesis),
fairds-2 should:
  (i) reduce the per-sample weight on majority samples whose gradient is
      aligned with the spurious axis,
  (ii) recover minority-group accuracy that vanilla MLP collapses on.

We compare three methods (vanilla, fairds-1, fairds-2) across:
  - majority_ratio ∈ {0.5, 0.7, 0.9, 0.99}
  - 5 seeds
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
from utils.seed import set_seed  # noqa: E402

from data import SpuriousConfig, make_balanced_validation, make_dataset  # noqa: E402


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, g):
        self.x, self.y, self.g = x, y, g

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i, self.g[i]


def run_one(
    *,
    method: str,
    majority_ratio: float,
    seed: int,
    epochs: int,
    n_total: int,
    n_val: int,
    batch_size: int,
    lr: float,
    alpha: float,
    hidden: int,
    spurious_strength: float,
    device: str,
) -> dict:
    set_seed(seed)
    cfg = SpuriousConfig(
        n_total=n_total,
        majority_ratio=majority_ratio,
        spurious_strength=spurious_strength,
        seed=seed,
    )
    x, y, g = make_dataset(cfg)
    x_val, y_val, g_val = make_balanced_validation(cfg, n_val=n_val)
    in_dim = x.shape[1]

    x = x.to(device); y = y.to(device); g = g.to(device)
    x_val = x_val.to(device); y_val = y_val.to(device); g_val = g_val.to(device)

    ds = IndexedTensorDataset(x, y, g)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = MLP(in_dim=in_dim, hidden=hidden)

    if method == "vanilla":
        log = train_vanilla(model, loader, x_val, y_val, epochs=epochs, lr=lr,
                            device=device, train_x=x, train_y=y)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, x_val, y_val, n_train=len(ds),
                           train_groups=g, order=order, alpha=alpha,
                           epochs=epochs, lr=lr, device=device,
                           train_x=x, train_y=y)
    else:
        raise ValueError(method)

    model.eval()
    with torch.no_grad():
        train_pred = model(x).argmax(-1)
        val_pred = model(x_val).argmax(-1)

    train_acc_by_group = {}
    val_acc_by_group = {}
    val_dp_by_group = {}
    for gid in (0, 1):
        m_tr = (g == gid)
        m_va = (g_val == gid)
        if m_tr.any():
            train_acc_by_group[gid] = float((train_pred[m_tr] == y[m_tr]).float().mean().item())
        if m_va.any():
            val_acc_by_group[gid] = float((val_pred[m_va] == y_val[m_va]).float().mean().item())
            val_dp_by_group[gid] = float(val_pred[m_va].float().mean().item())

    dp_diff = abs(val_dp_by_group.get(0, 0.0) - val_dp_by_group.get(1, 0.0))
    eo_pieces = {}
    for gid in (0, 1):
        m = (g_val == gid) & (y_val == 1)
        if m.any():
            eo_pieces[gid] = float(val_pred[m].float().mean().item())
    eo_diff = abs(eo_pieces.get(0, 0.0) - eo_pieces.get(1, 0.0))

    # Worst-group accuracy is THE metric for spurious-correlation tests.
    val_acc_worst = min(val_acc_by_group.values()) if val_acc_by_group else None

    phi_per_group_last = {}
    weight_per_group_last = {}
    if log.phi_per_group:
        last = log.phi_per_group[-1]
        for k, vs in last.items():
            phi_per_group_last[int(k)] = {
                "mean": float(np.mean(vs)),
                "median": float(np.median(vs)),
                "std": float(np.std(vs)),
                "n": int(len(vs)),
                "values": [float(v) for v in vs[:200]],
            }
    if log.weight_per_group:
        last_w = log.weight_per_group[-1]
        for k, vs in last_w.items():
            weight_per_group_last[int(k)] = {
                "mean": float(np.mean(vs)),
                "median": float(np.median(vs)),
                "std": float(np.std(vs)),
            }

    return {
        "method": method,
        "majority_ratio": majority_ratio,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "alpha": alpha,
        "hidden": hidden,
        "spurious_strength": spurious_strength,
        "final_val_acc": log.val_acc[-1] if log.val_acc else None,
        "final_val_acc_worst": val_acc_worst,
        "train_acc_by_group": {str(k): v for k, v in train_acc_by_group.items()},
        "val_acc_by_group": {str(k): v for k, v in val_acc_by_group.items()},
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "phi_per_group_last": phi_per_group_last,
        "weight_per_group_last": weight_per_group_last,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-total", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--spurious-strength", type=float, default=3.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.7, 0.9, 0.99])
    parser.add_argument("--methods", type=str, nargs="+", default=["vanilla", "fairds-1", "fairds-2"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", type=str, default="results/e1b")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e1b] writing to {out_dir}")

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
                    method=method,
                    majority_ratio=ratio,
                    seed=seed,
                    epochs=args.epochs,
                    n_total=args.n_total,
                    n_val=args.n_val,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    alpha=args.alpha,
                    hidden=args.hidden,
                    spurious_strength=args.spurious_strength,
                    device=args.device,
                )
                rec["walltime_sec"] = time.time() - tic
                runs.append(rec)
                worst = rec.get("final_val_acc_worst") or float("nan")
                print(
                    f"[{counter:>3}/{total}] {method:<10} ratio={ratio:<5} seed={seed} "
                    f"acc={rec['final_val_acc']:.3f} worst={worst:.3f} dp={rec['dp_diff']:.3f} "
                    f"({rec['walltime_sec']:.1f}s)"
                )

    out_file = out_dir / "sweep_results.json"
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0}, f, indent=2)
    print(f"[e1b] {out_file}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
