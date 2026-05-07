"""Waterbirds (Sagawa et al. 2020) — standard real-image spurious-correlation benchmark.

Setup
-----
- Task: bird type classification (waterbird vs landbird).
- Spurious feature: background (water vs land).
- Group = y * 2 + place ∈ {0,1,2,3}:
    g=0  landbird-on-land    (majority, easy)
    g=1  landbird-on-water   (minority, spurious flip)
    g=2  waterbird-on-land   (minority, spurious flip)
    g=3  waterbird-on-water  (majority, easy)
- Worst-group accuracy is the canonical metric.
- Standard ResNet-50 backbone (we use ResNet-18 for compute speed; the
  benchmark conclusions transfer).

Anchor (used inside training algorithms): subsample of val set,
**rebalanced to be group-balanced** (50 examples per group, 200 total).
Held-out val_eval: rest of the val split.
Test set: the original test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


DATA_ROOT = Path.home() / ".cache" / "fairds_data" / "waterbirds" / "waterbird_complete95_forest2water2"


@dataclass
class WaterbirdsConfig:
    n_anchor_per_group: int = 50
    image_size: int = 224
    seed: int = 0


class WaterbirdsImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = DATA_ROOT / row["img_filename"]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = int(row["y"])
        g = int(row["group"])
        return x, y, i, g


def _build_split_dfs(seed: int = 0, n_anchor_per_group: int = 50):
    md_path = DATA_ROOT / "metadata.csv"
    df = pd.read_csv(md_path)
    df["group"] = df["y"] * 2 + df["place"]

    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df = df[df["split"] == 1].reset_index(drop=True)
    test_df = df[df["split"] == 2].reset_index(drop=True)

    # Anchor: balanced per-group sample from val.
    rng = np.random.default_rng(seed)
    anchor_rows = []
    val_remaining = []
    for gid in (0, 1, 2, 3):
        idx_g = val_df.index[val_df["group"] == gid].to_numpy()
        rng.shuffle(idx_g)
        anchor_rows.extend(idx_g[:n_anchor_per_group].tolist())
        val_remaining.extend(idx_g[n_anchor_per_group:].tolist())
    anchor_df = val_df.iloc[anchor_rows].reset_index(drop=True)
    val_eval_df = val_df.iloc[val_remaining].reset_index(drop=True)
    return train_df, anchor_df, val_eval_df, test_df


def _transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_waterbirds(cfg: WaterbirdsConfig):
    train_df, anchor_df, val_eval_df, test_df = _build_split_dfs(
        seed=cfg.seed, n_anchor_per_group=cfg.n_anchor_per_group
    )
    train_tx = _transforms(cfg.image_size, train=True)
    eval_tx = _transforms(cfg.image_size, train=False)
    return {
        "train": WaterbirdsImageDataset(train_df, train_tx),
        "anchor": WaterbirdsImageDataset(anchor_df, eval_tx),
        "val_eval": WaterbirdsImageDataset(val_eval_df, eval_tx),
        "test": WaterbirdsImageDataset(test_df, eval_tx),
        "train_df": train_df,
        "anchor_df": anchor_df,
        "val_eval_df": val_eval_df,
        "test_df": test_df,
    }


def stack_dataset(ds: WaterbirdsImageDataset, device: str = "cpu") -> Tuple[Tensor, Tensor, Tensor]:
    """Materialize the entire dataset (e.g., the small anchor) as 3 tensors.
    Useful when an algorithm wants Xa/ya/ga as plain tensors."""
    xs, ys, gs = [], [], []
    for i in range(len(ds)):
        x, y, _idx, g = ds[i]
        xs.append(x)
        ys.append(y)
        gs.append(g)
    X = torch.stack(xs).to(device)
    Y = torch.tensor(ys, dtype=torch.long, device=device)
    G = torch.tensor(gs, dtype=torch.long, device=device)
    return X, Y, G


if __name__ == "__main__":
    cfg = WaterbirdsConfig()
    sets = make_waterbirds(cfg)
    for name in ("train", "anchor", "val_eval", "test"):
        df = sets[f"{name}_df"]
        print(name, "n=", len(df), "groups:", df["group"].value_counts().to_dict())
