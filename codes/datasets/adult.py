"""UCI Adult dataset loader for fairness experiments.

Sensitive attribute: sex (Male=0, Female=1). Target: income > 50K (binary).

Returns numpy arrays (X, y, g) plus a stratified train/val split helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CACHE_DIR = Path.home() / ".cache" / "fairds_data"


@dataclass
class AdultBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    g_val: np.ndarray
    feature_names: list[str]


def _encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals; return frame and resulting feature names."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=np.float32)
    return df, num_cols + [c for c in df.columns if c not in num_cols]


def load_adult(seed: int = 0, val_frac: float = 0.2) -> AdultBundle:
    """Loads Adult via OpenML (cached). Sensitive: sex. Target: income>50K."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "adult_bundle.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return AdultBundle(
            X_train=d["X_train"], y_train=d["y_train"], g_train=d["g_train"],
            X_val=d["X_val"], y_val=d["y_val"], g_val=d["g_val"],
            feature_names=d["feature_names"].tolist(),
        )

    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target

    # Drop rows with missing values (≈3% of rows)
    mask = ~df.isna().any(axis=1)
    df = df[mask].reset_index(drop=True)
    target = target[mask].reset_index(drop=True)

    # Sensitive attribute = sex
    sex = (df["sex"].astype(str).str.strip() == "Female").astype(np.int64).values
    df = df.drop(columns=["sex"])  # remove from features

    # One-hot encode categoricals
    encoded, feature_names = _encode(df)
    feature_names = encoded.columns.tolist()
    X = encoded.values.astype(np.float32)

    # Standardize numeric (also harmless on 0/1 dummies)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    y = (target.astype(str).str.strip() == ">50K").astype(np.int64).values

    X_tr, X_va, y_tr, y_va, g_tr, g_va = train_test_split(
        X, y, sex,
        test_size=val_frac, random_state=seed, stratify=np.stack([y, sex], axis=1).sum(axis=1)
    )

    bundle = AdultBundle(
        X_train=X_tr.astype(np.float32), y_train=y_tr.astype(np.int64), g_train=g_tr.astype(np.int64),
        X_val=X_va.astype(np.float32), y_val=y_va.astype(np.int64), g_val=g_va.astype(np.int64),
        feature_names=feature_names,
    )

    np.savez(cache,
             X_train=bundle.X_train, y_train=bundle.y_train, g_train=bundle.g_train,
             X_val=bundle.X_val, y_val=bundle.y_val, g_val=bundle.g_val,
             feature_names=np.array(bundle.feature_names, dtype=object))
    return bundle


def make_balanced_validation(bundle: AdultBundle, n_per_group: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample the val set into a 50:50 sex-balanced anchor of size 2*n_per_group."""
    rng = np.random.default_rng(seed)
    g = bundle.g_val
    out_idx = []
    for gid in (0, 1):
        idx = np.where(g == gid)[0]
        rng.shuffle(idx)
        out_idx.append(idx[:n_per_group])
    out_idx = np.concatenate(out_idx)
    rng.shuffle(out_idx)
    return bundle.X_val[out_idx], bundle.y_val[out_idx], bundle.g_val[out_idx]


if __name__ == "__main__":
    b = load_adult()
    print("X_train:", b.X_train.shape, "y_train:", b.y_train.shape)
    print("group dist train:", np.bincount(b.g_train), "val:", np.bincount(b.g_val))
    print("y dist train:", np.bincount(b.y_train), "val:", np.bincount(b.y_val))
    Xv, yv, gv = make_balanced_validation(b, n_per_group=200, seed=0)
    print("balanced val:", Xv.shape, np.bincount(gv))
