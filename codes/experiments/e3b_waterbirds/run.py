"""E3b — Waterbirds (Sagawa 2020) head-to-head sweep.

Real-image spurious-correlation benchmark per Codex Round 6 recommendation
(GPT-5.5 xhigh: 'less-synthetic spurious benchmark required for main-track').

Architecture: pretrained ResNet-18, BatchNorm running stats frozen so
torch.func.vmap can compute per-sample gradients without batch coupling.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairds.trainer import train_fairds, train_vanilla  # noqa: E402
from baselines.ren2018 import train_ren2018  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from datasets.waterbirds import WaterbirdsConfig, make_waterbirds, stack_dataset  # noqa: E402


def build_model(n_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with BN running stats frozen for vmap compatibility."""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    # Freeze BN running stats so vmap-per-sample-grad doesn't trip on
    # batch coupling. The affine params (weight/bias) remain trainable.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.track_running_stats = False
    return model


def evaluate(model, X, y, g, device, batch_size=256, num_groups=4):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            xb = X[i:i+batch_size].to(device)
            preds.append(model(xb).argmax(-1).cpu())
    pred = torch.cat(preds)
    y = y.cpu(); g = g.cpu()
    out = {"acc": float((pred == y).float().mean().item())}
    group_accs = []
    for gid in range(num_groups):
        m = (g == gid)
        if m.any():
            v = float((pred[m] == y[m]).float().mean().item())
            out[f"acc_g{gid}"] = v
            group_accs.append(v)
    out["worst_acc"] = min(group_accs) if group_accs else None
    return out


def materialize_anchor_to_device(anchor_ds, device):
    return stack_dataset(anchor_ds, device=device)


class IndexedDS(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, _idx, g = self.base[i]
        return x, y, i, g


def run_one(*, method: str, seed: int, epochs: int, batch_size: int, lr: float,
            alpha: float, temperature: float, weight_scale: float,
            n_anchor_per_group: int, image_size: int,
            device: str, num_workers: int, warmup_epochs: int = 0,
            no_ema: bool = False) -> dict:
    set_seed(seed)
    cfg = WaterbirdsConfig(n_anchor_per_group=n_anchor_per_group, image_size=image_size, seed=seed)
    sets = make_waterbirds(cfg)

    train_ds = IndexedDS(sets["train"])
    anchor_ds = sets["anchor"]
    val_eval_ds = sets["val_eval"]
    test_ds = sets["test"]

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=False)

    Xa, ya, ga = materialize_anchor_to_device(anchor_ds, device)
    Xv, yv, gv = materialize_anchor_to_device(val_eval_ds, device="cpu")  # large; eval-only
    Xte, yte, gte = materialize_anchor_to_device(test_ds, device="cpu")

    # train_groups arg expects per-sample group; we approximate with int tensor for n_train
    n_train = len(train_ds)
    train_groups = torch.tensor(sets["train_df"]["group"].values, dtype=torch.long).to(device)

    model = build_model(n_classes=2, pretrained=True).to(device)

    if method == "vanilla":
        log = train_vanilla(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    elif method.startswith("fairds-"):
        order = 1 if method.endswith("1") else 2
        log = train_fairds(model, loader, Xa, ya, n_train=n_train,
                           train_groups=train_groups, order=order, alpha=alpha,
                           temperature=temperature, weight_scale=weight_scale,
                           warmup_epochs=warmup_epochs, no_ema=no_ema,
                           epochs=epochs, lr=lr, device=device)
    elif method == "ren2018":
        log = train_ren2018(model, loader, Xa, ya, epochs=epochs, lr=lr, device=device)
    else:
        raise ValueError(method)

    val_metrics = evaluate(model, Xv, yv, gv, device)
    test_metrics = evaluate(model, Xte, yte, gte, device)

    rec = {
        "method": method, "seed": seed,
        "epochs": epochs, "lr": lr, "alpha": alpha,
        "temperature": temperature, "weight_scale": weight_scale,
        "image_size": image_size,
        "val_eval_acc": val_metrics["acc"],
        "val_eval_worst": val_metrics.get("worst_acc"),
        "test_acc": test_metrics["acc"],
        "test_worst": test_metrics.get("worst_acc"),
    }
    for gid in range(4):
        rec[f"val_eval_acc_g{gid}"] = val_metrics.get(f"acc_g{gid}")
        rec[f"test_acc_g{gid}"] = test_metrics.get(f"acc_g{gid}")
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["vanilla", "fairds-1", "fairds-2", "ren2018"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--weight-scale", type=float, default=4.0)
    p.add_argument("--n-anchor-per-group", type=int, default=50)
    p.add_argument("--image-size", type=int, default=96)  # smaller default for speed
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-root", type=str, default="results/e3b")
    args = p.parse_args()

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e3b] writing to {out_dir}")

    runs = []
    total = len(args.methods) * len(args.seeds)
    counter = 0
    t0 = time.time()
    for method in args.methods:
        for seed in args.seeds:
            counter += 1
            tic = time.time()
            try:
                rec = run_one(
                    method=method, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, alpha=args.alpha, temperature=args.temperature,
                    weight_scale=args.weight_scale, n_anchor_per_group=args.n_anchor_per_group,
                    image_size=args.image_size, device=args.device, num_workers=args.num_workers,
                    warmup_epochs=args.warmup_epochs, no_ema=args.no_ema,
                )
                rec["walltime_sec"] = time.time() - tic
                runs.append(rec)
                print(f"[{counter:>3}/{total}] {method:<10} seed={seed} "
                      f"val_acc={rec['val_eval_acc']:.3f} val_worst={rec['val_eval_worst']:.3f} "
                      f"test_acc={rec['test_acc']:.3f} test_worst={rec['test_worst']:.3f} "
                      f"({rec['walltime_sec']:.1f}s)")
            except Exception as e:
                print(f"[{counter}/{total}] {method} seed={seed} FAILED: {e}")
                import traceback; traceback.print_exc()

    out_file = out_dir / "sweep_results.json"
    out_file.write_text(json.dumps({"args": vars(args), "runs": runs, "total_walltime_sec": time.time()-t0}, indent=2))
    print(f"[e3b] {out_file}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
