"""Synthetic 2-group Gaussian mixture for E1.

Each example carries a (binary) class label y in {0,1} and a (binary)
group attribute g in {0,1}. We construct distributions so that:
  - ground-truth decision boundary is the same for both groups
  - group sizes are imbalanced according to a configurable ratio
  - features for the minority group are slightly noisier so that an
    unweighted classifier overfits the majority distribution and shows
    measurable group-level performance gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class ToyConfig:
    n_total: int = 2000
    majority_ratio: float = 0.9  # 0.9 means 90:10 split
    feature_dim: int = 20
    noise_majority: float = 0.5
    noise_minority: float = 1.0
    class_separation: float = 1.5
    seed: int = 0


def make_dataset(cfg: ToyConfig) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (x, y, g) tensors. x: (n, d). y: (n,) in {0,1}. g: (n,) in {0,1}."""

    rng = np.random.default_rng(cfg.seed)
    n_majority = int(cfg.n_total * cfg.majority_ratio)
    n_minority = cfg.n_total - n_majority

    def sample_group(n: int, noise: float, group_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Two classes per group, separated along the first 5 dims.
        n_pos = n // 2
        n_neg = n - n_pos
        mean_pos = np.zeros(cfg.feature_dim)
        mean_pos[:5] = +cfg.class_separation / 2
        mean_neg = np.zeros(cfg.feature_dim)
        mean_neg[:5] = -cfg.class_separation / 2
        x_pos = rng.normal(mean_pos, noise, size=(n_pos, cfg.feature_dim))
        x_neg = rng.normal(mean_neg, noise, size=(n_neg, cfg.feature_dim))
        x = np.concatenate([x_pos, x_neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
        g = np.full(n, group_id, dtype=np.int64)
        order = rng.permutation(n)
        return x[order], y[order], g[order]

    x_maj, y_maj, g_maj = sample_group(n_majority, cfg.noise_majority, 0)
    x_min, y_min, g_min = sample_group(n_minority, cfg.noise_minority, 1)

    x = np.concatenate([x_maj, x_min], axis=0)
    y = np.concatenate([y_maj, y_min], axis=0)
    g = np.concatenate([g_maj, g_min], axis=0)

    order = rng.permutation(len(x))
    x, y, g = x[order], y[order], g[order]

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(g),
    )


def make_balanced_validation(
    cfg: ToyConfig, n_val: int = 400
) -> Tuple[Tensor, Tensor, Tensor]:
    """Validation set with 50:50 group composition for the unbiased anchor.

    We re-use the same generative process but with majority_ratio=0.5 and
    a separate seed offset to avoid overlap with training.
    """
    val_cfg = ToyConfig(
        n_total=n_val,
        majority_ratio=0.5,
        feature_dim=cfg.feature_dim,
        noise_majority=cfg.noise_majority,
        noise_minority=cfg.noise_minority,
        class_separation=cfg.class_separation,
        seed=cfg.seed + 1_000_000,
    )
    return make_dataset(val_cfg)
