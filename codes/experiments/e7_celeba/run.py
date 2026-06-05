"""E7 — CelebA fairness (Blond_Hair target, Male sensitive group).

Demographic + image-structured bias: tests whether the closed-form Shapley
reweighter mitigates demographic bias when the bias is structured (image), in
contrast to the diffuse tabular Adult/COMPAS where it was weak. Reports worst-group
accuracy (4 groups), the majority-minority disparity, and the gender DP/EO gaps.

Architecture: from-scratch 3-conv CNN (64 -> 8), our strength regime.
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
from utils.seed import set_seed  # noqa: E402
from datasets.celeba_fairness import CelebAConfig, make_celeba_fairness  # noqa: E402


class CelebaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
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


def evaluate(model, X, y, g, male, device, batch_size=256, num_groups=4):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            preds.append(model(X[i:i + batch_size].to(device)).argmax(-1).cpu())
    pred = torch.cat(preds); y = y.cpu(); g = g.cpu(); male = male.cpu()
    out = {"acc": float((pred == y).float().mean().item())}
    group_accs = []
    for gid in range(num_groups):
        m = (g == gid)
        if m.any():
            v = float((pred[m] == y[m]).float().mean().item())
            out[f"acc_g{gid}"] = v
            group_accs.append(v)
    out["worst_acc"] = min(group_accs) if group_accs else None
    out["disparity"] = (max(group_accs) - min(group_accs)) if group_accs else None
    # gender DP / EO gaps
    dp, eo = {}, {}
    for mid in (0, 1):
        mm = (male == mid)
        if mm.any():
            dp[mid] = float(pred[mm].float().mean().item())
        my = mm & (y == 1)
        if my.any():
            eo[mid] = float(pred[my].float().mean().item())
    out["dp_diff"] = abs(dp.get(0, 0.0) - dp.get(1, 0.0))
    out["eo_diff"] = abs(eo.get(0, 0.0) - eo.get(1, 0.0))
    return out


def run_one(*, method, seed, epochs, batch_size, lr, alpha, device,
            temperature=0.1, weight_scale=4.0, arm=None) -> dict:
    set_seed(seed)
    b = make_celeba_fairness(CelebAConfig(seed=seed))
    Xt = b.X_train.to(device); yt = b.y_train.to(device); gt = b.g_train.to(device)
    Xa = b.X_anchor.to(device); ya = b.y_anchor.to(device)
    Xv, yv, gv, mv = b.X_val_eval, b.y_val_eval, b.g_val_eval, b.male_val_eval
    Xte, yte, gte, mte = b.X_test, b.y_test, b.g_test, b.male_test

    ds = IndexedDS(Xt, yt, gt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = CelebaCNN()

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
        log = train_groupdro(model, loader, Xa, ya, n_groups=4, epochs=epochs, lr=lr, device=device)
    else:
        raise ValueError(method)

    val = evaluate(model, Xv, yv, gv, mv, device)
    test = evaluate(model, Xte, yte, gte, mte, device)
    rec = {
        "method": method, "seed": seed, "arm": arm, "epochs": epochs, "lr": lr,
        "alpha": alpha, "temperature": temperature, "weight_scale": weight_scale,
        "val_eval_acc": val["acc"], "val_eval_worst": val.get("worst_acc"),
        "val_eval_disparity": val.get("disparity"),
        "test_acc": test["acc"], "test_worst": test.get("worst_acc"),
        "test_disparity": test.get("disparity"),
        "test_dp_diff": test.get("dp_diff"), "test_eo_diff": test.get("eo_diff"),
    }
    for gid in range(4):
        rec[f"test_acc_g{gid}"] = test.get(f"acc_g{gid}")
    if hasattr(log, "weight_std_per_batch") and log.weight_std_per_batch:
        rec["w_std_mean"] = float(np.mean(log.weight_std_per_batch))
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["vanilla", "fairds-1", "fairds-2", "ren2018"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--weight-scale", type=float, default=4.0)
    p.add_argument("--arm", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e7_celeba")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e7] writing to {out_dir}")

    runs = []
    total = len(args.methods) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for method in args.methods:
        for seed in args.seeds:
            counter += 1
            tic = time.time()
            try:
                rec = run_one(method=method, seed=seed, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr, alpha=args.alpha,
                              temperature=args.temperature, weight_scale=args.weight_scale,
                              device=args.device, arm=args.arm)
                rec["walltime_sec"] = time.time() - tic
                runs.append(rec)
                print(f"[{counter:>2}/{total}] {method:<10} seed={seed} "
                      f"test_worst={rec['test_worst']:.3f} disp={rec['test_disparity']:.3f} "
                      f"dp={rec['test_dp_diff']:.3f} eo={rec['test_eo_diff']:.3f} "
                      f"acc={rec['test_acc']:.3f} ({rec['walltime_sec']:.1f}s)")
            except Exception as e:
                print(f"[{counter}/{total}] {method} seed={seed} FAILED: {e}")
                import traceback; traceback.print_exc()

    out = out_dir / "sweep_results.json"
    out.write_text(json.dumps({"args": vars(args), "runs": runs, "total_walltime_sec": time.time() - t0}, indent=2))
    print(f"[e7] {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
