"""Spurious-correlation 2-group dataset for E1b.

Goal: produce a TRUE representation-bias setup where Vanilla relies on a
spurious feature that holds in the majority group but not the minority,
so that group-1 accuracy collapses unless reweighting attenuates the
spurious shortcut.

Design
------
- Each sample has features (x_core, z_spurious) ∈ R^{d_core + 1}.
- x_core ~ N(class_mean, 1.0), where class_mean differs between y=0 and y=1
  along the first `d_core_signal` dimensions. Same conditional dist for
  both groups.
- z_spurious:
    * majority group (g=0): perfectly correlated with y (y=1 → z ~ N(+s, 0.3),
      y=0 → z ~ N(-s, 0.3)) with strength s.
    * minority group (g=1): independent of y, z ~ N(0, 0.3).
- A model that exploits z gets perfect accuracy on majority but ~50% on
  minority (representation bias).

Validation set
--------------
50:50 balanced across groups, drawn from the same distributions as
training but with a separate seed offset. Because half of the val set
is the minority (where z is uninformative), a model that only relies
on z gets ~75% val acc — leaving room for fairds to improve via x_core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class SpuriousConfig:
    n_total: int = 2000
    majority_ratio: float = 0.9
    d_core: int = 8
    d_core_signal: int = 2  # only 2 of 8 core dims carry the *true* signal
    spurious_strength: float = 10.0  # strong spurious axis
    spurious_noise: float = 0.3
    class_separation: float = 0.5  # weak true signal -> spurious dominates
    seed: int = 0


def _sample_group(
    n: int, group_id: int, cfg: SpuriousConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pos = n // 2
    n_neg = n - n_pos

    # x_core
    mean_pos = np.zeros(cfg.d_core)
    mean_pos[: cfg.d_core_signal] = +cfg.class_separation / 2
    mean_neg = np.zeros(cfg.d_core)
    mean_neg[: cfg.d_core_signal] = -cfg.class_separation / 2

    x_pos = rng.normal(mean_pos, 1.0, size=(n_pos, cfg.d_core))
    x_neg = rng.normal(mean_neg, 1.0, size=(n_neg, cfg.d_core))

    # z_spurious
    if group_id == 0:
        z_pos = rng.normal(+cfg.spurious_strength, cfg.spurious_noise, size=n_pos)
        z_neg = rng.normal(-cfg.spurious_strength, cfg.spurious_noise, size=n_neg)
    else:
        z_pos = rng.normal(0.0, cfg.spurious_noise, size=n_pos)
        z_neg = rng.normal(0.0, cfg.spurious_noise, size=n_neg)

    feat_pos = np.concatenate([x_pos, z_pos[:, None]], axis=1)
    feat_neg = np.concatenate([x_neg, z_neg[:, None]], axis=1)

    feats = np.concatenate([feat_pos, feat_neg], axis=0).astype(np.float32)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    groups = np.full(n, group_id, dtype=np.int64)

    order = rng.permutation(n)
    return feats[order], labels[order], groups[order]


def make_dataset(cfg: SpuriousConfig) -> Tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(cfg.seed)
    n_majority = int(cfg.n_total * cfg.majority_ratio)
    n_minority = cfg.n_total - n_majority

    x_maj, y_maj, g_maj = _sample_group(n_majority, 0, cfg, rng)
    x_min, y_min, g_min = _sample_group(n_minority, 1, cfg, rng)

    x = np.concatenate([x_maj, x_min], axis=0)
    y = np.concatenate([y_maj, y_min], axis=0)
    g = np.concatenate([g_maj, g_min], axis=0)
    order = rng.permutation(len(x))
    return (
        torch.from_numpy(x[order]),
        torch.from_numpy(y[order]),
        torch.from_numpy(g[order]),
    )


def make_balanced_validation(
    cfg: SpuriousConfig, n_val: int = 400
) -> Tuple[Tensor, Tensor, Tensor]:
    val_cfg = SpuriousConfig(
        n_total=n_val,
        majority_ratio=0.5,  # 50:50 balanced anchor
        d_core=cfg.d_core,
        d_core_signal=cfg.d_core_signal,
        spurious_strength=cfg.spurious_strength,
        spurious_noise=cfg.spurious_noise,
        class_separation=cfg.class_separation,
        seed=cfg.seed + 1_000_000,
    )
    return make_dataset(val_cfg)
