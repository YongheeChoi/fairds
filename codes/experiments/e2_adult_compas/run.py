"""E2 — Adult / COMPAS head-to-head sweep.

Tests claims C2 (Fairds vs FORML on real data) and C4 (sensitive-attribute-free
validation set). For this first pass we report against vanilla baseline with
two D_val composition modes:
  - balanced: 50:50 sex/race balanced anchor (the canonical Fairds setting)
  - random : random subsample of val (no stratification, simulates the
    no-sensitive-label regime — Fairds still works because phi uses only
    plain val loss).

Models:
  - Adult: 2-layer MLP (hidden=64)
  - COMPAS: logistic regression
(Per plan.md.)

Methods: vanilla, fairds-1, fairds-2. FORML / FairShap / ARL baselines are
added later if Fairds shows a positive fairness signal.
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
from utils.seed import set_seed  # noqa: E402
from datasets.adult import load_adult, make_balanced_validation as adult_balanced_val  # noqa: E402
from datasets.compas import load_compas, make_balanced_validation as compas_balanced_val  # noqa: E402


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class LinearLR(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, g):
        self.x, self.y, self.g = x, y, g

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i, self.g[i]


def get_data(dataset: str, seed: int, n_val_per_group: int, val_mode: str):
    if dataset == "adult":
        bundle = load_adult(seed=seed)
        if val_mode == "balanced":
            Xv, yv, gv = adult_balanced_val(bundle, n_per_group=n_val_per_group, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            n = 2 * n_val_per_group
            idx = rng.choice(len(bundle.X_val), size=min(n, len(bundle.X_val)), replace=False)
            Xv, yv, gv = bundle.X_val[idx], bundle.y_val[idx], bundle.g_val[idx]
        return bundle.X_train, bundle.y_train, bundle.g_train, Xv, yv, gv
    elif dataset == "compas":
        bundle = load_compas(seed=seed)
        if val_mode == "balanced":
            Xv, yv, gv = compas_balanced_val(bundle, n_per_group=n_val_per_group, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            n = 2 * n_val_per_group
            idx = rng.choice(len(bundle.X_val), size=min(n, len(bundle.X_val)), replace=False)
            Xv, yv, gv = bundle.X_val[idx], bundle.y_val[idx], bundle.g_val[idx]
        return bundle.X_train, bundle.y_train, bundle.g_train, Xv, yv, gv
    raise ValueError(dataset)


def evaluate_full_val(model, bundle_or_eval_xyg, device):
    """Evaluate on the FULL val set (not the small anchor)."""
    Xv, yv, gv = bundle_or_eval_xyg
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(Xv).to(device)
        y = torch.from_numpy(yv).to(device)
        g = torch.from_numpy(gv).to(device)
        pred = model(x).argmax(-1)
    out = {"val_acc": float((pred == y).float().mean().item())}
    for gid in (0, 1):
        m = (g == gid)
        if m.any():
            out[f"val_acc_g{gid}"] = float((pred[m] == y[m]).float().mean().item())
            out[f"val_dp_g{gid}"] = float(pred[m].float().mean().item())
    out["dp_diff"] = abs(out.get("val_dp_g0", 0.0) - out.get("val_dp_g1", 0.0))
    eo = {}
    for gid in (0, 1):
        m = (g == gid) & (y == 1)
        if m.any():
            eo[gid] = float(pred[m].float().mean().item())
    out["eo_diff"] = abs(eo.get(0, 0.0) - eo.get(1, 0.0))
    out["worst_acc"] = min(out.get(f"val_acc_g{gid}", 1.0) for gid in (0, 1))
    return out


def run_one(*, dataset: str, method: str, seed: int, val_mode: str,
            epochs: int, batch_size: int, lr: float, alpha: float,
            n_val_per_group: int, device: str,
            temperature: float = 0.5, clip_quantile: float = 0.05,
            weight_scale: float = 1.0) -> dict:
    set_seed(seed)
    Xt, yt, gt, Xv_anchor, yv_anchor, gv_anchor = get_data(dataset, seed, n_val_per_group, val_mode)

    in_dim = Xt.shape[1]
    if dataset == "adult":
        model = MLP(in_dim=in_dim, hidden=64)
    else:
        model = LinearLR(in_dim=in_dim)

    x_t = torch.from_numpy(Xt).to(device)
    y_t = torch.from_numpy(yt).to(device)
    g_t = torch.from_numpy(gt).to(device)
    x_va = torch.from_numpy(Xv_anchor).to(device)
    y_va = torch.from_numpy(yv_anchor).to(device)

    ds = IndexedTensorDataset(x_t, y_t, g_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if method == "vanilla":
        log = train_vanilla(model, loader, x_va, y_va, epochs=epochs, lr=lr, device=device,
                            train_x=x_t, train_y=y_t)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, x_va, y_va, n_train=len(ds),
                           train_groups=g_t, order=order, alpha=alpha,
                           temperature=temperature, clip_quantile=clip_quantile,
                           weight_scale=weight_scale,
                           epochs=epochs, lr=lr, device=device,
                           train_x=x_t, train_y=y_t)
    elif method == "ren2018":
        log = train_ren2018(model, loader, x_va, y_va, epochs=epochs, lr=lr,
                            device=device, train_x=x_t, train_y=y_t)
    else:
        raise ValueError(method)

    # Full val eval (use the bundle's full val set, not the anchor)
    if dataset == "adult":
        b = load_adult(seed=seed)
    else:
        b = load_compas(seed=seed)
    metrics = evaluate_full_val(model, (b.X_val, b.y_val, b.g_val), device)

    rec = {
        "dataset": dataset, "method": method, "seed": seed, "val_mode": val_mode,
        "epochs": epochs, "lr": lr, "alpha": alpha, "n_val_per_group": n_val_per_group,
        "temperature": temperature, "clip_quantile": clip_quantile, "weight_scale": weight_scale,
        "val_acc": metrics["val_acc"], "worst_acc": metrics["worst_acc"],
        "val_acc_g0": metrics.get("val_acc_g0"), "val_acc_g1": metrics.get("val_acc_g1"),
        "dp_diff": metrics["dp_diff"], "eo_diff": metrics["eo_diff"],
    }
    # Diagnostic magnitudes (Round 1 fix)
    if hasattr(log, "weight_std_per_batch") and log.weight_std_per_batch:
        import numpy as _np
        rec["w_std_mean"] = float(_np.mean(log.weight_std_per_batch))
        rec["w_max_mean"] = float(_np.mean(log.weight_max_per_batch))
        rec["w_min_mean"] = float(_np.mean(log.weight_min_per_batch))
        rec["phi_std_mean"] = float(_np.mean(log.phi_std_per_batch))
    # Per-group phi summary at last epoch (only for fairds methods)
    phi_summ = {}
    if hasattr(log, "phi_per_group") and log.phi_per_group:
        last = log.phi_per_group[-1]
        for k, vs in last.items():
            phi_summ[int(k)] = {
                "mean": float(np.mean(vs)),
                "std": float(np.std(vs)),
                "n": int(len(vs)),
            }
    rec["phi_per_group_last"] = phi_summ
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["adult", "compas"])
    p.add_argument("--methods", nargs="+", default=["vanilla", "fairds-1", "fairds-2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--val-modes", nargs="+", default=["balanced", "random"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-val-per-group", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e2")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e2] writing to {out_dir}")

    runs = []
    total = len(args.datasets) * len(args.methods) * len(args.val_modes) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for ds in args.datasets:
        for method in args.methods:
            for vm in args.val_modes:
                for seed in args.seeds:
                    counter += 1
                    tic = time.time()
                    try:
                        rec = run_one(
                            dataset=ds, method=method, seed=seed, val_mode=vm,
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, alpha=args.alpha,
                            n_val_per_group=args.n_val_per_group, device=args.device,
                        )
                        rec["walltime_sec"] = time.time() - tic
                        runs.append(rec)
                        print(
                            f"[{counter:>3}/{total}] {ds:<7} {method:<10} val={vm:<8} seed={seed} "
                            f"acc={rec['val_acc']:.3f} worst={rec['worst_acc']:.3f} "
                            f"dp={rec['dp_diff']:.3f} eo={rec['eo_diff']:.3f} ({rec['walltime_sec']:.1f}s)"
                        )
                    except Exception as e:
                        print(f"[{counter:>3}/{total}] {ds} {method} {vm} seed={seed} FAILED: {e}")

    out_file = out_dir / "sweep_results.json"
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0}, f, indent=2)
    print(f"[e2] {out_file}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
