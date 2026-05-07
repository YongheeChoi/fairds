"""Colored MNIST (Arjovsky et al. 2019, IRM) — standard spurious-correlation benchmark.

Setup
-----
- Take MNIST. Label y = 1 if digit >= 5 else 0 (binary task).
- For each example, define a *color* c that is spuriously correlated with y.
  - In training environment, P(c = y) = (1 - p_flip).  We use two training
    environments with different flip rates to mimic IRM's setup, but for a
    Fairds-style "majority vs minority group" framing we collapse them into:
    - "majority" group (g=0): color c agrees with label y with prob 0.9
      (strong spurious correlation, like train env 1).
    - "minority" group (g=1): color c agrees with label y with prob 0.5
      (no spurious correlation, like a clean group).
  - At test time: P(c = y) = 0.1, flipped from the majority training group.
- Color realization: red channel = 1.0 if c=1 else 0.0; green channel
  similarly with c=0; blue channel zeroed out.

Train majority_ratio defaults to 0.9 (matches the controlled E1b setup).

A vanilla CNN trained on this should latch onto color (the easy spurious
feature) and collapse on the test set. Worst-group accuracy is the
primary metric — we report majority test acc, minority test acc, and
their gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms


CACHE_DIR = Path.home() / ".cache" / "fairds_data" / "mnist"


@dataclass
class CMNISTConfig:
    n_train: int = 5000          # subset of MNIST for speed
    n_anchor: int = 200          # 50:50 group-balanced anchor used INSIDE training (D_val)
    n_val_eval: int = 500        # held-out, NOT used by any algorithm during training
    n_test: int = 5000
    majority_ratio: float = 0.9
    p_color_majority: float = 0.9   # P(c = y | g=0)
    p_color_minority: float = 0.5   # P(c = y | g=1)
    p_color_test: float = 0.1       # P(c = y | test)
    seed: int = 0


@dataclass
class CMNISTBundle:
    X_train: Tensor   # (n, 2, 28, 28)
    y_train: Tensor
    g_train: Tensor
    X_anchor: Tensor  # 50:50 balanced anchor used INSIDE training algorithms
    y_anchor: Tensor
    g_anchor: Tensor
    X_val_eval: Tensor  # 50:50 balanced HELD-OUT eval set, never seen by any algorithm
    y_val_eval: Tensor
    g_val_eval: Tensor
    X_test: Tensor    # test env, color flipped (OOD)
    y_test: Tensor
    g_test: Tensor    # group on test = whether color matches label (spurious-aligned vs spurious-flipped)


def _load_mnist():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tx = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(str(CACHE_DIR), train=True, download=True, transform=tx)
    test = datasets.MNIST(str(CACHE_DIR), train=False, download=True, transform=tx)
    Xtr = train.data.float() / 255.0  # (60000, 28, 28)
    ytr = train.targets
    Xte = test.data.float() / 255.0
    yte = test.targets
    return Xtr.numpy(), ytr.numpy(), Xte.numpy(), yte.numpy()


def _colorize(x_gray, c):
    """x_gray: (n, 28, 28). c: (n,) in {0, 1}. Return (n, 2, 28, 28)."""
    n = x_gray.shape[0]
    out = np.zeros((n, 2, 28, 28), dtype=np.float32)
    for i in range(n):
        if c[i] == 1:
            out[i, 0] = x_gray[i]   # red channel
        else:
            out[i, 1] = x_gray[i]   # green channel
    return out


def _make_balanced_set(Xraw, yraw_bin, n_per_group, p_color_majority,
                       p_color_minority, rng) -> tuple:
    """Build a 50:50 group-balanced colored set (used for anchor + val_eval)."""
    idx_pool = rng.choice(len(Xraw), size=2 * n_per_group, replace=False)
    Xg = Xraw[idx_pool]
    yg = yraw_bin[idx_pool]
    g = np.concatenate([
        np.zeros(n_per_group, dtype=np.int64),
        np.ones(n_per_group, dtype=np.int64),
    ])
    rng.shuffle(g)
    flip_maj = rng.random(2 * n_per_group) > p_color_majority
    flip_min = rng.random(2 * n_per_group) > p_color_minority
    flip = np.where(g == 0, flip_maj, flip_min)
    color = np.where(flip, 1 - yg, yg).astype(np.int64)
    return _colorize(Xg, color), yg, g


def make_colored_mnist(cfg: CMNISTConfig) -> CMNISTBundle:
    rng = np.random.default_rng(cfg.seed)
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = _load_mnist()

    # Binary label
    y_train_bin = (ytr_raw >= 5).astype(np.int64)
    y_test_bin = (yte_raw >= 5).astype(np.int64)

    # Disjoint subsamples for train, anchor, val_eval (all from MNIST train split)
    needed = cfg.n_train + cfg.n_anchor + cfg.n_val_eval
    pool = rng.choice(len(Xtr_raw), size=needed, replace=False)
    idx_tr = pool[: cfg.n_train]
    idx_anchor = pool[cfg.n_train : cfg.n_train + cfg.n_anchor]
    idx_eval = pool[cfg.n_train + cfg.n_anchor : needed]

    # Train group assignment
    n_majority = int(cfg.n_train * cfg.majority_ratio)
    n_minority = cfg.n_train - n_majority
    g_train = np.concatenate([
        np.zeros(n_majority, dtype=np.int64),
        np.ones(n_minority, dtype=np.int64),
    ])
    rng.shuffle(g_train)

    Xtr_gray = Xtr_raw[idx_tr]
    ytr = y_train_bin[idx_tr]
    flip_majority = rng.random(cfg.n_train) > cfg.p_color_majority
    flip_minority = rng.random(cfg.n_train) > cfg.p_color_minority
    flip = np.where(g_train == 0, flip_majority, flip_minority)
    color_tr = np.where(flip, 1 - ytr, ytr).astype(np.int64)
    Xtr_color = _colorize(Xtr_gray, color_tr)

    # Anchor (used inside algorithms): 50:50 from idx_anchor
    n_anchor_per_group = cfg.n_anchor // 2
    Xa_pool = Xtr_raw[idx_anchor]
    ya_pool = y_train_bin[idx_anchor]
    ga = np.concatenate([
        np.zeros(n_anchor_per_group, dtype=np.int64),
        np.ones(cfg.n_anchor - n_anchor_per_group, dtype=np.int64),
    ])
    rng.shuffle(ga)
    flip_a_maj = rng.random(cfg.n_anchor) > cfg.p_color_majority
    flip_a_min = rng.random(cfg.n_anchor) > cfg.p_color_minority
    flip_a = np.where(ga == 0, flip_a_maj, flip_a_min)
    color_a = np.where(flip_a, 1 - ya_pool, ya_pool).astype(np.int64)
    Xa_color = _colorize(Xa_pool, color_a)

    # Val_eval (HELD-OUT, never seen by any algorithm during training): also
    # 50:50 group-balanced for fair worst-group eval.
    n_eval_per_group = cfg.n_val_eval // 2
    Xe_pool = Xtr_raw[idx_eval]
    ye_pool = y_train_bin[idx_eval]
    ge = np.concatenate([
        np.zeros(n_eval_per_group, dtype=np.int64),
        np.ones(cfg.n_val_eval - n_eval_per_group, dtype=np.int64),
    ])
    rng.shuffle(ge)
    flip_e_maj = rng.random(cfg.n_val_eval) > cfg.p_color_majority
    flip_e_min = rng.random(cfg.n_val_eval) > cfg.p_color_minority
    flip_e = np.where(ge == 0, flip_e_maj, flip_e_min)
    color_e = np.where(flip_e, 1 - ye_pool, ye_pool).astype(np.int64)
    Xe_color = _colorize(Xe_pool, color_e)

    # Test: spurious flipped. Test groups = (color matches label) vs not.
    idx_te = rng.choice(len(Xte_raw), size=cfg.n_test, replace=False)
    Xte_gray = Xte_raw[idx_te]
    yte = y_test_bin[idx_te]
    flip_te = rng.random(cfg.n_test) > cfg.p_color_test
    color_te = np.where(flip_te, 1 - yte, yte).astype(np.int64)
    Xte_color = _colorize(Xte_gray, color_te)
    # group_test: 0 = color matches label (spurious-aligned, easy for ERM),
    #             1 = color disagrees (spurious-flipped, hard for ERM).
    g_test = (color_te != yte).astype(np.int64)

    return CMNISTBundle(
        X_train=torch.from_numpy(Xtr_color).float(),
        y_train=torch.from_numpy(ytr).long(),
        g_train=torch.from_numpy(g_train).long(),
        X_anchor=torch.from_numpy(Xa_color).float(),
        y_anchor=torch.from_numpy(ya_pool).long(),
        g_anchor=torch.from_numpy(ga).long(),
        X_val_eval=torch.from_numpy(Xe_color).float(),
        y_val_eval=torch.from_numpy(ye_pool).long(),
        g_val_eval=torch.from_numpy(ge).long(),
        X_test=torch.from_numpy(Xte_color).float(),
        y_test=torch.from_numpy(yte).long(),
        g_test=torch.from_numpy(g_test).long(),
    )


if __name__ == "__main__":
    b = make_colored_mnist(CMNISTConfig())
    print("X_train:", b.X_train.shape, "y dist:", b.y_train.unique(return_counts=True))
    print("group dist train:", b.g_train.unique(return_counts=True))
    print("X_anchor:", b.X_anchor.shape, "group dist:", b.g_anchor.unique(return_counts=True))
    print("X_val_eval:", b.X_val_eval.shape, "group dist:", b.g_val_eval.unique(return_counts=True))
    print("X_test:", b.X_test.shape, "test groups (aligned/flipped):", b.g_test.unique(return_counts=True))
