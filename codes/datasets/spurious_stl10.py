"""Spurious-STL10 (binary) — texture-corruption spurious benchmark on 96px natural images.

The THIRD spurious regime in this project, after:
  - Colored-MNIST     (color spurious,   28px synthetic digits)
  - Corrupted-CIFAR   (texture spurious, 32px small natural images)
Spurious-STL10 keeps the *texture-corruption* spurious cue of Corrupted-CIFAR but
moves to larger, more naturalistic 96px STL-10 photographs — a genuinely distinct
data distribution, to test whether Fairds's 2nd-order mechanism transfers a third time.

Spurious cue = corruption TYPE (color-independent, like Corrupted-CIFAR):
  - corruption 0: Gaussian blur            (low-pass / smooth texture)
  - corruption 1: additive Gaussian noise  (high-frequency / grainy texture)

Binary task on car (STL class 2) vs truck (class 9): both vehicles, so the shape cue
is weak and a from-scratch CNN latches onto the easy corruption texture, then collapses
on the spurious-flipped OOD test set. Headline metric = OOD worst-group (spurious-flipped)
test accuracy, mirroring Colored-MNIST and Corrupted-CIFAR exactly.

STL-10 binary (car+truck) has only 2600 images, so we pool the STL train+test splits and
draw four DISJOINT subsets (train / anchor / val_eval / test). OOD is enforced by the
corruption injection schedule (p_corr_test=0.1), identical to Corrupted-CIFAR — the test
images are disjoint from train, and the spurious cue is flipped relative to the majority
group, so corruption-reliant models fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision.datasets import STL10


CACHE_DIR = Path.home() / ".cache" / "fairds_data" / "stl10"


@dataclass
class SpuriousSTL10Config:
    class_a: int = 2             # car   -> label 0
    class_b: int = 9             # truck -> label 1 (both vehicles: shape cue is hard,
    #                              forcing the model onto the easy corruption cue)
    n_train: int = 1400
    n_anchor: int = 200          # 50:50 group-balanced anchor used INSIDE training (D_val)
    n_val_eval: int = 300        # held-out, NOT used by any algorithm during training
    n_test: int = 600
    majority_ratio: float = 0.9
    p_corr_majority: float = 0.9   # P(corruption = label | g=0)
    p_corr_minority: float = 0.5   # P(corruption = label | g=1)
    p_corr_test: float = 0.1       # P(corruption = label | test)
    blur_kernel: int = 9           # larger kernel for 96px (vs 5 for 32px CIFAR)
    blur_sigma: float = 2.0
    noise_std: float = 0.12
    seed: int = 0


@dataclass
class SpuriousSTL10Bundle:
    X_train: Tensor   # (n, 3, 96, 96)
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


def _load_stl():
    """STL-10 train+test pooled. STL10.data is already (N,3,96,96) uint8 (CHW)."""
    tr = STL10(str(CACHE_DIR), split="train", download=False)
    te = STL10(str(CACHE_DIR), split="test", download=False)
    X = np.concatenate([tr.data, te.data]).astype(np.float32) / 255.0  # (13000,3,96,96)
    y = np.concatenate([tr.labels, te.labels]).astype(np.int64)
    return X, y


def _binarize(X, y, class_a, class_b):
    mask = (y == class_a) | (y == class_b)
    return X[mask], (y[mask] == class_b).astype(np.int64)


def _corrupt(X, ctype, cfg, rng):
    """Apply per-sample corruption. X: (n,3,96,96) np float; ctype: (n,) {0,1}."""
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    out = Xt.clone()
    m0 = torch.from_numpy(ctype == 0)
    m1 = torch.from_numpy(ctype == 1)
    if m0.any():
        out[m0] = TF.gaussian_blur(Xt[m0], kernel_size=int(cfg.blur_kernel),
                                   sigma=float(cfg.blur_sigma))
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


def _balanced_groups(n, rng):
    g = np.concatenate([np.zeros(n // 2, dtype=np.int64),
                        np.ones(n - n // 2, dtype=np.int64)])
    rng.shuffle(g)
    return g


def make_spurious_stl10(cfg: SpuriousSTL10Config) -> SpuriousSTL10Bundle:
    rng = np.random.default_rng(cfg.seed)
    X_raw, y_raw = _load_stl()
    X_raw, y_bin = _binarize(X_raw, y_raw, cfg.class_a, cfg.class_b)

    # Four DISJOINT subsets from the pooled binary set (only ~2600 images total).
    needed = cfg.n_train + cfg.n_anchor + cfg.n_val_eval + cfg.n_test
    if needed > len(X_raw):
        raise ValueError(f"need {needed} but only {len(X_raw)} car+truck images")
    pool = rng.choice(len(X_raw), size=needed, replace=False)
    i_tr = pool[: cfg.n_train]
    i_an = pool[cfg.n_train: cfg.n_train + cfg.n_anchor]
    i_ev = pool[cfg.n_train + cfg.n_anchor: cfg.n_train + cfg.n_anchor + cfg.n_val_eval]
    i_te = pool[cfg.n_train + cfg.n_anchor + cfg.n_val_eval: needed]

    # --- Train: majority/minority split with spurious corruption ---
    n_majority = int(cfg.n_train * cfg.majority_ratio)
    g_train = np.concatenate([np.zeros(n_majority, dtype=np.int64),
                              np.ones(cfg.n_train - n_majority, dtype=np.int64)])
    rng.shuffle(g_train)
    ytr = y_bin[i_tr]
    ctype_tr = _assign_corruption(ytr, g_train, cfg.p_corr_majority, cfg.p_corr_minority, rng)
    Xtr = _corrupt(X_raw[i_tr], ctype_tr, cfg, rng)

    # --- Anchor (used INSIDE algorithms): 50:50 group-balanced ---
    ya = y_bin[i_an]
    ga = _balanced_groups(cfg.n_anchor, rng)
    ctype_a = _assign_corruption(ya, ga, cfg.p_corr_majority, cfg.p_corr_minority, rng)
    Xa = _corrupt(X_raw[i_an], ctype_a, cfg, rng)

    # --- Val_eval (HELD-OUT, never seen during training): 50:50 balanced ---
    ye = y_bin[i_ev]
    ge = _balanced_groups(cfg.n_val_eval, rng)
    ctype_e = _assign_corruption(ye, ge, cfg.p_corr_majority, cfg.p_corr_minority, rng)
    Xe = _corrupt(X_raw[i_ev], ctype_e, cfg, rng)

    # --- Test: spurious flipped. groups = (corruption matches label) vs not. ---
    yte = y_bin[i_te]
    flip_te = rng.random(len(i_te)) > cfg.p_corr_test
    ctype_te = np.where(flip_te, 1 - yte, yte).astype(np.int64)
    Xte = _corrupt(X_raw[i_te], ctype_te, cfg, rng)
    g_test = (ctype_te != yte).astype(np.int64)

    def T(a, dt): return torch.from_numpy(a).to(dt)
    return SpuriousSTL10Bundle(
        X_train=T(Xtr, torch.float32), y_train=T(ytr, torch.long), g_train=T(g_train, torch.long),
        X_anchor=T(Xa, torch.float32), y_anchor=T(ya, torch.long), g_anchor=T(ga, torch.long),
        X_val_eval=T(Xe, torch.float32), y_val_eval=T(ye, torch.long), g_val_eval=T(ge, torch.long),
        X_test=T(Xte, torch.float32), y_test=T(yte, torch.long), g_test=T(g_test, torch.long),
    )


if __name__ == "__main__":
    b = make_spurious_stl10(SpuriousSTL10Config())
    print("X_train:", b.X_train.shape, "y dist:", b.y_train.unique(return_counts=True))
    print("group dist train:", b.g_train.unique(return_counts=True))
    print("X_anchor:", b.X_anchor.shape, "group dist:", b.g_anchor.unique(return_counts=True))
    print("X_test:", b.X_test.shape, "test groups (aligned/flipped):", b.g_test.unique(return_counts=True))
