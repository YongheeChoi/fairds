"""ProPublica COMPAS recidivism dataset loader.

We follow the standard ProPublica filter:
  - days_b_screening_arrest in [-30, 30]
  - is_recid != -1
  - c_charge_degree != "O"
  - score_text != "N/A"

Sensitive attribute: race (Black=1 vs Caucasian=0). Target: two_year_recid.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CACHE_DIR = Path.home() / ".cache" / "fairds_data"
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"


@dataclass
class CompasBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    g_val: np.ndarray
    feature_names: list[str]


def _download_csv() -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = CACHE_DIR / "compas_two_years.csv"
    if raw.exists():
        return raw.read_bytes()
    with urllib.request.urlopen(COMPAS_URL, timeout=30) as r:
        data = r.read()
    raw.write_bytes(data)
    return data


def load_compas(seed: int = 0, val_frac: float = 0.2) -> CompasBundle:
    cache = CACHE_DIR / "compas_bundle.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return CompasBundle(
            X_train=d["X_train"], y_train=d["y_train"], g_train=d["g_train"],
            X_val=d["X_val"], y_val=d["y_val"], g_val=d["g_val"],
            feature_names=d["feature_names"].tolist(),
        )

    raw = _download_csv()
    df = pd.read_csv(io.BytesIO(raw))

    # ProPublica standard filter
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    # Restrict to Black vs Caucasian (the standard ProPublica fairness pair)
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()

    feat_cols = [
        "sex", "age", "age_cat", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree",
    ]
    df_feat = df[feat_cols].copy()
    df_feat = pd.get_dummies(df_feat, columns=["sex", "age_cat", "c_charge_degree"], drop_first=False, dtype=np.float32)
    feature_names = df_feat.columns.tolist()
    X = df_feat.values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    g = (df["race"].values == "African-American").astype(np.int64)  # 1 = Black, 0 = Caucasian
    y = df["two_year_recid"].values.astype(np.int64)

    X_tr, X_va, y_tr, y_va, g_tr, g_va = train_test_split(
        X, y, g,
        test_size=val_frac, random_state=seed, stratify=np.stack([y, g], axis=1).sum(axis=1),
    )

    bundle = CompasBundle(
        X_train=X_tr, y_train=y_tr, g_train=g_tr,
        X_val=X_va, y_val=y_va, g_val=g_va,
        feature_names=feature_names,
    )
    np.savez(cache,
             X_train=bundle.X_train, y_train=bundle.y_train, g_train=bundle.g_train,
             X_val=bundle.X_val, y_val=bundle.y_val, g_val=bundle.g_val,
             feature_names=np.array(bundle.feature_names, dtype=object))
    return bundle


def make_balanced_validation(bundle: CompasBundle, n_per_group: int = 100, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = bundle.g_val
    out = []
    for gid in (0, 1):
        idx = np.where(g == gid)[0]
        rng.shuffle(idx)
        out.append(idx[: min(n_per_group, len(idx))])
    out_idx = np.concatenate(out)
    rng.shuffle(out_idx)
    return bundle.X_val[out_idx], bundle.y_val[out_idx], bundle.g_val[out_idx]


if __name__ == "__main__":
    b = load_compas()
    print("X_train:", b.X_train.shape, "y_train:", b.y_train.shape)
    print("group dist train:", np.bincount(b.g_train), "val:", np.bincount(b.g_val))
    print("y dist train:", np.bincount(b.y_train), "val:", np.bincount(b.y_val))
    Xv, yv, gv = make_balanced_validation(b, n_per_group=100, seed=0)
    print("balanced val:", Xv.shape, np.bincount(gv))
