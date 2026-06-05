"""CelebA fairness benchmark — Blond_Hair target with Male as the sensitive group.

A demographic AND image-structured bias benchmark (Sagawa et al.\ 2020 style). In
CelebA, blond hair is strongly correlated with being female, so an ERM model uses
gender to predict blond and fails the minority **blond-male** group. Unlike the
diffuse tabular demographic data (Adult/COMPAS), here the bias is *structured*
(one dominant attribute in image space) --- the regime where the closed-form
Shapley reweighter is expected to work. We report worst-group accuracy and the
gender DP/EO gaps.

Groups: g = 2*blond + male  (0: nonblond-F, 1: nonblond-M, 2: blond-F, 3: blond-M);
the worst group is blond-male. We keep `male` separately for DP/EO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision.datasets import CelebA

CACHE = Path.home() / ".cache" / "fairds_data" / "celeba"


@dataclass
class CelebAConfig:
    image_size: int = 64
    n_train: int = 10000      # natural distribution (blond-male is the rare minority)
    n_anchor: int = 200       # 4-group balanced, used inside algorithms
    n_val_eval: int = 400     # 4-group balanced, held-out
    n_test: int = 2000        # 4-group balanced (so worst-group has enough samples)
    seed: int = 0


@dataclass
class CelebABundle:
    X_train: Tensor; y_train: Tensor; g_train: Tensor; male_train: Tensor
    X_anchor: Tensor; y_anchor: Tensor; g_anchor: Tensor
    X_val_eval: Tensor; y_val_eval: Tensor; g_val_eval: Tensor; male_val_eval: Tensor
    X_test: Tensor; y_test: Tensor; g_test: Tensor; male_test: Tensor


def _load_split(split: str):
    ds = CelebA(str(CACHE), split=split, target_type="attr", download=False)
    bi = ds.attr_names.index("Blond_Hair")
    mi = ds.attr_names.index("Male")
    y = ds.attr[:, bi].long()
    male = ds.attr[:, mi].long()
    g = (2 * y + male).long()
    return ds, y, male, g


def _load_images(ds, idx, size):
    imgs = [TF.to_tensor(ds[int(i)][0].resize((size, size))) for i in idx]
    return torch.stack(imgs)


def _balanced_per_group(g_np, rng, per_group, exclude=None):
    """Return concatenated indices with `per_group` from each of the 4 groups."""
    out = []
    for gid in range(4):
        gi = np.where(g_np == gid)[0]
        if exclude is not None:
            gi = gi[~np.isin(gi, exclude)]
        gi = rng.permutation(gi)
        out.append(gi[:per_group])
    return np.concatenate(out)


def make_celeba_fairness(cfg: CelebAConfig) -> CelebABundle:
    rng = np.random.default_rng(cfg.seed)
    tr_ds, ytr, maletr, gtr = _load_split("train")
    te_ds, yte, malete, gte = _load_split("test")
    gtr_np = gtr.numpy()

    # Balanced anchor + held-out val_eval from the train split
    anchor_idx = _balanced_per_group(gtr_np, rng, cfg.n_anchor // 4)
    val_idx = _balanced_per_group(gtr_np, rng, cfg.n_val_eval // 4, exclude=anchor_idx)
    used = np.concatenate([anchor_idx, val_idx])

    # Train: NATURAL distribution (keeps the blond<->female bias), disjoint from anchor/val
    avail = np.setdiff1d(np.arange(len(ytr)), used, assume_unique=False)
    tr_idx = rng.choice(avail, min(cfg.n_train, len(avail)), replace=False)

    # Test: 4-group balanced so the worst (blond-male) group is well sampled
    test_idx = _balanced_per_group(gte.numpy(), rng, cfg.n_test // 4)

    Xt = _load_images(tr_ds, tr_idx, cfg.image_size)
    Xa = _load_images(tr_ds, anchor_idx, cfg.image_size)
    Xv = _load_images(tr_ds, val_idx, cfg.image_size)
    Xte = _load_images(te_ds, test_idx, cfg.image_size)

    return CelebABundle(
        X_train=Xt, y_train=ytr[tr_idx], g_train=gtr[tr_idx], male_train=maletr[tr_idx],
        X_anchor=Xa, y_anchor=ytr[anchor_idx], g_anchor=gtr[anchor_idx],
        X_val_eval=Xv, y_val_eval=ytr[val_idx], g_val_eval=gtr[val_idx], male_val_eval=maletr[val_idx],
        X_test=Xte, y_test=yte[test_idx], g_test=gte[test_idx], male_test=malete[test_idx],
    )


if __name__ == "__main__":
    b = make_celeba_fairness(CelebAConfig())
    print("X_train:", b.X_train.shape, "blond rate:", float(b.y_train.float().mean()))
    print("train group dist:", torch.bincount(b.g_train, minlength=4).tolist(),
          "(0 nonblondF,1 nonblondM,2 blondF,3 blondM)")
    print("anchor group dist:", torch.bincount(b.g_anchor, minlength=4).tolist())
    print("test group dist:", torch.bincount(b.g_test, minlength=4).tolist())
