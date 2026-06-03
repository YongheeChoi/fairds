"""Corrupted-CIFAR-10 (binary) — a texture-corruption spurious benchmark.

A *color-independent* sibling of Colored MNIST. Instead of a spurious COLOR
channel, the spurious cue is the corruption TYPE applied to the image:

- corruption 0: Gaussian blur     (low-pass / smooth texture)
- corruption 1: additive Gaussian noise (high-frequency / grainy texture)

Binary task on two CIFAR-10 classes. As in Colored MNIST:
- "majority" group (g=0): corruption agrees with label with prob 0.9
  (strong spurious correlation).
- "minority" group (g=1): corruption agrees with label with prob 0.5
  (no spurious correlation).
- test: P(corruption = label) = 0.1, flipped from the majority group.

A from-scratch CNN latches onto the easy corruption-texture cue and
collapses on the test set. The headline metric is worst-group
(spurious-flipped) test accuracy, mirroring Colored MNIST exactly so the
two benchmarks form an apples-to-apples pair across two distinct spurious
modalities (color vs texture).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision import datasets


CACHE_DIR = Path.home() / ".cache" / "fairds_data" / "cifar10"


@dataclass
class CorruptedCIFARConfig:
    class_a: int = 1             # automobile (label 0)
    class_b: int = 9            # truck      (label 1)  — both vehicles, shape cue is hard,
    #                              forcing the model toward the easy corruption cue
    n_train: int = 5000
    n_anchor: int = 200          # 50:50 group-balanced anchor used INSIDE training (D_val)
    n_val_eval: int = 500        # held-out, NOT used by any algorithm during training
    n_test: int = 2000
    majority_ratio: float = 0.9
    p_corr_majority: float = 0.9   # P(corruption = label | g=0)
    p_corr_minority: float = 0.5   # P(corruption = label | g=1)
    p_corr_test: float = 0.1       # P(corruption = label | test)
    blur_sigma: float = 1.2
    noise_std: float = 0.12
    seed: int = 0


@dataclass
class CorruptedCIFARBundle:
    X_train: Tensor   # (n, 3, 32, 32)
    y_train: Tensor
    g_train: Tensor
    X_anchor: Tensor
    y_anchor: Tensor
    g_anchor: Tensor
    X_val_eval: Tensor
    y_val_eval: Tensor
    g_val_eval: Tensor
    X_test: Tensor
    y_test: Tensor
    g_test: Tensor    # 0 = corruption matches label (spurious-aligned), 1 = flipped


def _load_cifar():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(str(CACHE_DIR), train=True, download=True)
    test = datasets.CIFAR10(str(CACHE_DIR), train=False, download=True)

    def to_chw(X):  # (N,32,32,3) uint8 -> (N,3,32,32) float in [0,1]
        return (X.transpose(0, 3, 1, 2).astype(np.float32)) / 255.0

    return (to_chw(train.data), np.array(train.targets),
            to_chw(test.data), np.array(test.targets))


def _binarize(X, y, class_a, class_b):
    mask = (y == class_a) | (y == class_b)
    return X[mask], (y[mask] == class_b).astype(np.int64)


def _corrupt(X, ctype, cfg, rng):
    """Apply per-sample corruption. X: (n,3,32,32) np float; ctype: (n,) {0,1}."""
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    out = Xt.clone()
    m0 = torch.from_numpy(ctype == 0)
    m1 = torch.from_numpy(ctype == 1)
    if m0.any():
        out[m0] = TF.gaussian_blur(Xt[m0], kernel_size=5, sigma=float(cfg.blur_sigma))
    if m1.any():
        sh = tuple(out[m1].shape)
        noise = torch.from_numpy(rng.normal(0.0, cfg.noise_std, size=sh).astype(np.float32))
        out[m1] = (Xt[m1] + noise).clamp(0.0, 1.0)
    return out.numpy()


def _assign_corruption(y, g, p_maj, p_min, rng):
    """corruption_type = y with prob p (per group), else 1-y (spurious flip)."""
    n = len(y)
    flip_maj = rng.random(n) > p_maj
    flip_min = rng.random(n) > p_min
    flip = np.where(g == 0, flip_maj, flip_min)
    return np.where(flip, 1 - y, y).astype(np.int64)


def make_corrupted_cifar(cfg: CorruptedCIFARConfig) -> CorruptedCIFARBundle:
    rng = np.random.default_rng(cfg.seed)
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = _load_cifar()
    Xtr_raw, ytr_bin = _binarize(Xtr_raw, ytr_raw, cfg.class_a, cfg.class_b)
    Xte_raw, yte_bin = _binarize(Xte_raw, yte_raw, cfg.class_a, cfg.class_b)

    # Disjoint subsamples for train, anchor, val_eval (all from CIFAR train split)
    needed = cfg.n_train + cfg.n_anchor + cfg.n_val_eval
    pool = rng.choice(len(Xtr_raw), size=needed, replace=False)
    idx_tr = pool[: cfg.n_train]
    idx_anchor = pool[cfg.n_train: cfg.n_train + cfg.n_anchor]
    idx_eval = pool[cfg.n_train + cfg.n_anchor: needed]

    # --- Train: majority/minority split with spurious corruption ---
    n_majority = int(cfg.n_train * cfg.majority_ratio)
    g_train = np.concatenate([
        np.zeros(n_majority, dtype=np.int64),
        np.ones(cfg.n_train - n_majority, dtype=np.int64),
    ])
    rng.shuffle(g_train)
    ytr = ytr_bin[idx_tr]
    ctype_tr = _assign_corruption(ytr, g_train, cfg.p_corr_majority, cfg.p_corr_minority, rng)
    Xtr = _corrupt(Xtr_raw[idx_tr], ctype_tr, cfg, rng)

    # --- Anchor (used INSIDE algorithms): 50:50 group-balanced ---
    ya = ytr_bin[idx_anchor]
    ga = np.concatenate([
        np.zeros(cfg.n_anchor // 2, dtype=np.int64),
        np.ones(cfg.n_anchor - cfg.n_anchor // 2, dtype=np.int64),
    ])
    rng.shuffle(ga)
    ctype_a = _assign_corruption(ya, ga, cfg.p_corr_majority, cfg.p_corr_minority, rng)
    Xa = _corrupt(Xtr_raw[idx_anchor], ctype_a, cfg, rng)

    # --- Val_eval (HELD-OUT, never seen during training): 50:50 balanced ---
    ye = ytr_bin[idx_eval]
    ge = np.concatenate([
        np.zeros(cfg.n_val_eval // 2, dtype=np.int64),
        np.ones(cfg.n_val_eval - cfg.n_val_eval // 2, dtype=np.int64),
    ])
    rng.shuffle(ge)
    ctype_e = _assign_corruption(ye, ge, cfg.p_corr_majority, cfg.p_corr_minority, rng)
    Xe = _corrupt(Xtr_raw[idx_eval], ctype_e, cfg, rng)

    # --- Test: spurious flipped. groups = (corruption matches label) vs not. ---
    n_test = min(cfg.n_test, len(Xte_raw))
    idx_te = rng.choice(len(Xte_raw), size=n_test, replace=False)
    yte = yte_bin[idx_te]
    flip_te = rng.random(n_test) > cfg.p_corr_test
    ctype_te = np.where(flip_te, 1 - yte, yte).astype(np.int64)
    Xte = _corrupt(Xte_raw[idx_te], ctype_te, cfg, rng)
    g_test = (ctype_te != yte).astype(np.int64)

    def T(a, dt): return torch.from_numpy(a).to(dt)
    return CorruptedCIFARBundle(
        X_train=T(Xtr, torch.float32), y_train=T(ytr, torch.long), g_train=T(g_train, torch.long),
        X_anchor=T(Xa, torch.float32), y_anchor=T(ya, torch.long), g_anchor=T(ga, torch.long),
        X_val_eval=T(Xe, torch.float32), y_val_eval=T(ye, torch.long), g_val_eval=T(ge, torch.long),
        X_test=T(Xte, torch.float32), y_test=T(yte, torch.long), g_test=T(g_test, torch.long),
    )


if __name__ == "__main__":
    b = make_corrupted_cifar(CorruptedCIFARConfig())
    print("X_train:", b.X_train.shape, "y dist:", b.y_train.unique(return_counts=True))
    print("group dist train:", b.g_train.unique(return_counts=True))
    print("X_anchor:", b.X_anchor.shape, "group dist:", b.g_anchor.unique(return_counts=True))
    print("X_test:", b.X_test.shape, "test groups (aligned/flipped):", b.g_test.unique(return_counts=True))
