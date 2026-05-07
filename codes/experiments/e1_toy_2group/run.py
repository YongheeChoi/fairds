"""E1 — Toy 2-group logistic regression sweep.

Tests claim C3: under increasing group imbalance, the 2nd-order Fairds
Shapley value should become *lower* for majority-group samples than for
minority-group samples (Wilcoxon p < 0.05 at 90:10), and the gap should
grow monotonically with imbalance.

We compare three training methods on top of the same synthetic data:
  - vanilla SGD
  - Fairds-1st (gradient dot-product Shapley reweighting)
  - Fairds-2nd (+ gradient-Hessian-gradient cross-term)

Outputs go to results/e1/<timestamp>/sweep_results.json plus a per-run
record. Final weights and per-group Shapley distributions are recorded
for downstream Wilcoxon analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make codes/ importable when running as a script.
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairds.trainer import train_fairds, train_vanilla  # noqa: E402
from utils.seed import set_seed  # noqa: E402

from data import ToyConfig, make_balanced_validation, make_dataset  # noqa: E402


def build_model(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 2)


class IndexedTensorDataset(torch.utils.data.Dataset):
    """Yields (x, y, idx, group) tuples."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> None:
        self.x = x
        self.y = y
        self.g = g

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, i: int):
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
    device: str,
) -> dict:
    set_seed(seed)
    cfg = ToyConfig(
        n_total=n_total, majority_ratio=majority_ratio, seed=seed
    )
    x, y, g = make_dataset(cfg)
    x_val, y_val, g_val = make_balanced_validation(cfg, n_val=n_val)

    x = x.to(device)
    y = y.to(device)
    g = g.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    g_val = g_val.to(device)

    ds = IndexedTensorDataset(x, y, g)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = build_model(input_dim=cfg.feature_dim)

    if method == "vanilla":
        log = train_vanilla(
            model,
            loader,
            x_val,
            y_val,
            epochs=epochs,
            lr=lr,
            device=device,
            train_x=x,
            train_y=y,
        )
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(
            model,
            loader,
            x_val,
            y_val,
            n_train=len(ds),
            train_groups=g,
            order=order,
            alpha=alpha,
            epochs=epochs,
            lr=lr,
            device=device,
            train_x=x,
            train_y=y,
        )
    else:
        raise ValueError(method)

    # Group-level metrics on val and train at end of training.
    model.eval()
    with torch.no_grad():
        train_pred = model(x).argmax(-1)
        val_pred = model(x_val).argmax(-1)

    train_acc_by_group = {}
    val_acc_by_group = {}
    val_dp_by_group = {}
    for g_id in (0, 1):
        m_tr = (g == g_id)
        m_va = (g_val == g_id)
        if m_tr.any():
            train_acc_by_group[g_id] = float((train_pred[m_tr] == y[m_tr]).float().mean().item())
        if m_va.any():
            val_acc_by_group[g_id] = float((val_pred[m_va] == y_val[m_va]).float().mean().item())
            val_dp_by_group[g_id] = float(val_pred[m_va].float().mean().item())

    # DP-diff = |P(y_hat=1|g=0) - P(y_hat=1|g=1)|
    dp_diff = abs(val_dp_by_group.get(0, 0.0) - val_dp_by_group.get(1, 0.0))
    # EO-diff: condition on true positives.
    eo_pieces = {}
    for g_id in (0, 1):
        m = (g_val == g_id) & (y_val == 1)
        if m.any():
            eo_pieces[g_id] = float(val_pred[m].float().mean().item())
    eo_diff = abs(eo_pieces.get(0, 0.0) - eo_pieces.get(1, 0.0))

    # If Fairds: compute final Shapley summaries by group (last epoch).
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
                "values": [float(v) for v in vs[:200]],  # cap for storage
            }
    if log.weight_per_group:
        last_w = log.weight_per_group[-1]
        for k, vs in last_w.items():
            weight_per_group_last[int(k)] = {
                "mean": float(np.mean(vs)),
                "median": float(np.median(vs)),
                "std": float(np.std(vs)),
                "n": int(len(vs)),
            }

    record = {
        "method": method,
        "majority_ratio": majority_ratio,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "alpha": alpha,
        "final_train_acc": log.train_acc[-1] if log.train_acc else None,
        "final_val_acc": log.val_acc[-1] if log.val_acc else None,
        "final_val_loss": log.val_loss[-1] if log.val_loss else None,
        "train_acc_by_group": {str(k): v for k, v in train_acc_by_group.items()},
        "val_acc_by_group": {str(k): v for k, v in val_acc_by_group.items()},
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "phi_per_group_last": phi_per_group_last,
        "weight_per_group_last": weight_per_group_last,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-total", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9, 0.99],
        help="Majority-group share of training set.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["vanilla", "fairds-1", "fairds-2"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", type=str, default="results/e1")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e1] writing to {out_dir}")

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
                    device=args.device,
                )
                rec["walltime_sec"] = time.time() - tic
                runs.append(rec)
                print(
                    f"[{counter:>3}/{total}] {method:<10} ratio={ratio:<5} seed={seed} "
                    f"acc={rec['final_val_acc']:.3f} dp_diff={rec['dp_diff']:.3f} "
                    f"({rec['walltime_sec']:.1f}s)"
                )

    out_file = out_dir / "sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(
            {"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0},
            f,
            indent=2,
        )
    print(f"[e1] {out_file}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
