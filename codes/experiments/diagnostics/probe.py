"""Mechanism probes for fairds (answers "why" behind the empirical findings).

D2 (H2 — why 2nd-order is regime-dependent):
    Measure, in two regimes, the validation-Hessian spectrum and the
    geometry between the first-order term phi1_i = <g_i, g_val> and the
    cross-term c_i = <g_i, H_val g_val>:
      - from-scratch    : Corrupted-CIFAR CNN, full parameters
      - last-layer FT   : Waterbirds frozen ResNet-18, fc head only
    Prediction: from-scratch has a LARGE spectral gap and first-order is
    nearly ORTHOGONAL to the cross-term (cross adds new information ->
    2nd-order helps); last-layer has a SMALL gap and the cross-term is
    nearly PARALLEL to first-order (redundant -> 2nd-order barely moves).

D1 (H1 — why last-layer beats full-parameter in fine-tuning):
    In a full-parameter Waterbirds model, split g_i into backbone vs fc
    parts and measure each part's norm and its majority<->minority
    separation. Prediction: the fc part carries the group-discriminative
    signal; the backbone part is large-but-undiscriminative, drowning the
    fc signal in <g_i, g_val>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairds.shapley import (  # noqa: E402
    _per_sample_gradients, _validation_gradient, _hessian_vector_product,
)
from fairds.trainer import train_vanilla  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from datasets.corrupted_cifar import CorruptedCIFARConfig, make_corrupted_cifar  # noqa: E402
from datasets.waterbirds import WaterbirdsConfig, make_waterbirds, stack_dataset  # noqa: E402
from experiments.e5_corrupted_cifar.run import CifarCNN  # noqa: E402
from experiments.e3b_waterbirds.run import build_model  # noqa: E402


class _DS(Dataset):
    def __init__(self, x, y, g): self.x, self.y, self.g = x, y, g
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i], i, self.g[i]


def quick_train(model, xt, yt, gt, xv, yv, dev, epochs, lr, bs=128):
    ld = DataLoader(_DS(xt, yt, gt), batch_size=bs, shuffle=True)
    train_vanilla(model, ld, xv, yv, epochs=epochs, lr=lr, device=dev)


@torch.no_grad()
def _normalize(v):
    return v / v.norm().clamp_min(1e-12)


def top_eigs(model, xv, yv, k=3, n_iter=25):
    """Top-k eigenvalues of H_val by power iteration + deflation."""
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    dev = xv.device
    eigs, vecs = [], []
    for _ in range(k):
        v = _normalize(torch.randn(P, device=dev))
        for _ in range(n_iter):
            Hv = _hessian_vector_product(model, F.cross_entropy, xv, yv, v)
            for lam, u in zip(eigs, vecs):
                Hv = Hv - lam * (u @ v) * u
            v = _normalize(Hv)
        Hv = _hessian_vector_product(model, F.cross_entropy, xv, yv, v)
        eigs.append((v @ Hv).item())
        vecs.append(v.detach())
    return eigs


def probe_d2(model, xt, yt, gt, xv, yv, n_sub=300):
    eigs = top_eigs(model, xv, yv, k=3)
    g_val = _validation_gradient(model, F.cross_entropy, xv, yv)
    Hg = _hessian_vector_product(model, F.cross_entropy, xv, yv, g_val)
    idx = torch.randperm(len(xt))[:n_sub]
    g_i = _per_sample_gradients(model, F.cross_entropy, xt[idx], yt[idx])
    first = g_i @ g_val
    cross = g_i @ Hg
    # rms-normalize cross to first (exactly as the algorithm does)
    cn = cross * (first.pow(2).mean().sqrt() / cross.pow(2).mean().sqrt().clamp_min(1e-12))
    g = gt[idx]
    maj, mn = (g == 0), (g == 1)

    def ms(t):
        return float(t[maj].mean()), float(t[mn].mean())

    cos = float((first @ cross) / (first.norm() * cross.norm()).clamp_min(1e-12))
    # root of cos≈1: is H_val g_val parallel to g_val? (g_val ~ top eigenvector?)
    align_Hg = float((Hg @ g_val) / (Hg.norm() * g_val.norm()).clamp_min(1e-12))
    # effective per-group scaling that the cross-term applies to phi1
    phi2 = first - 0.5 * cn
    gap_first = ms(first)[1] - ms(first)[0]      # minority - majority on phi1
    gap_phi2 = ms(phi2)[1] - ms(phi2)[0]         # minority - majority on phi2
    abse = [abs(e) for e in eigs]
    return {
        "eigs": [round(e, 3) for e in eigs],
        "spectral_gap_l1_over_l2": round(abse[0] / max(abse[1], 1e-9), 2),
        "first_maj_min": tuple(round(x, 4) for x in ms(first)),
        "cross_maj_min": tuple(round(x, 4) for x in ms(cn)),
        "cos_first_cross": round(cos, 3),
        "align_Hgval_gval": round(align_Hg, 3),
        "min_minus_maj_phi1": round(gap_first, 3),
        "min_minus_maj_phi2": round(gap_phi2, 3),
        "n_sub": n_sub,
    }


def gap_vs_alpha(model, xt, yt, gt, xv, yv, alphas, n_sub=300):
    """Minority-majority phi gap as the 2nd-order weight alpha grows.
    H* predicts |gap| shrinks monotonically with alpha (curvature damping of
    the first-order reweighting). alpha=0 recovers fairds-1's gap."""
    g_val = _validation_gradient(model, F.cross_entropy, xv, yv)
    Hg = _hessian_vector_product(model, F.cross_entropy, xv, yv, g_val)
    idx = torch.randperm(len(xt))[:n_sub]
    g_i = _per_sample_gradients(model, F.cross_entropy, xt[idx], yt[idx])
    first = g_i @ g_val
    cross = g_i @ Hg
    cn = cross * (first.pow(2).mean().sqrt() / cross.pow(2).mean().sqrt().clamp_min(1e-12))
    g = gt[idx]
    maj, mn = (g == 0), (g == 1)
    out = {}
    for a in alphas:
        phi2 = first - a * cn
        out[a] = round(float(phi2[mn].mean() - phi2[maj].mean()), 3)
    return out


def probe_d1_perlayer(model, xt, yt, gt, xv, yv, n_sub=300):
    """Split per-sample grad into backbone vs fc; measure norm + group separation."""
    g_val = _validation_gradient(model, F.cross_entropy, xv, yv)
    # parameter index ranges
    names, sizes = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n); sizes.append(p.numel())
    offs = np.cumsum([0] + sizes)
    fc_mask = torch.zeros(int(offs[-1]), dtype=torch.bool, device=xv.device)
    for n, a, b in zip(names, offs[:-1], offs[1:]):
        if n.startswith("fc."):
            fc_mask[a:b] = True
    idx = torch.randperm(len(xt))[:n_sub]
    g_i = _per_sample_gradients(model, F.cross_entropy, xt[idx], yt[idx])
    g = gt[idx]
    maj, mn = (g == 0), (g == 1)

    def block_stats(mask):
        gb = g_i[:, mask]
        gvb = g_val[mask]
        norm = float(gb.norm(dim=1).mean())
        # majority vs minority mean-gradient cosine: low cosine => the block
        # separates the two groups (discriminative); high => undiscriminative
        gm, gn = gb[maj].mean(0), gb[mn].mean(0)
        cos = float((gm @ gn) / (gm.norm() * gn.norm()).clamp_min(1e-12))
        # contribution of this block to <g_i, g_val>
        contrib = float((gb @ gvb).abs().mean())
        return {"mean_grad_norm": round(norm, 4),
                "maj_min_cos": round(cos, 3),
                "abs_dotval_contrib": round(contrib, 4)}

    return {"backbone": block_stats(~fc_mask), "fc": block_stats(fc_mask)}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}\n" + "=" * 70)

    # ---------- Regime 1: FROM-SCRATCH (Corrupted-CIFAR, full params) ----------
    set_seed(0)
    b = make_corrupted_cifar(CorruptedCIFARConfig(seed=0))
    xt, yt, gt = b.X_train.to(dev), b.y_train.to(dev), b.g_train.to(dev)
    xa, ya = b.X_anchor.to(dev), b.y_anchor.to(dev)
    ga = b.g_anchor.to(dev)
    model = CifarCNN().to(dev)
    quick_train(model, xt, yt, gt, xa, ya, dev, epochs=8, lr=0.02)
    print("[D2] FROM-SCRATCH (Corrupted-CIFAR, full params)")
    print("   ", probe_d2(model, xt, yt, gt, xa, ya))
    print("[H* alpha-sweep] from-scratch min-maj phi gap vs alpha (expect |gap| monotone down):")
    print("   ", gap_vs_alpha(model, xt, yt, gt, xa, ya, [0.0, 0.25, 0.5, 1.0, 2.0]))

    # ---------- Regime 2: LAST-LAYER FT (Waterbirds frozen, fc only) ----------
    set_seed(0)
    cfg = WaterbirdsConfig(image_size=96, seed=0, n_anchor_per_group=50)
    sets = make_waterbirds(cfg)
    Xt, Yt, Gt = stack_dataset(sets["train"], device=dev)
    Xa, Ya, Ga = stack_dataset(sets["anchor"], device=dev)
    model = build_model(freeze_backbone=True).to(dev)
    quick_train(model, Xt, Yt, Gt, Xa, Ya, dev, epochs=10, lr=0.1, bs=128)
    print("\n[D2] LAST-LAYER FT (Waterbirds frozen ResNet-18, fc only)")
    print("   ", probe_d2(model, Xt, Yt, Gt, Xa, Ya))

    # ---------- D1: FULL-PARAM Waterbirds, backbone vs fc ----------
    set_seed(0)
    model = build_model(freeze_backbone=False).to(dev)
    quick_train(model, Xt, Yt, Gt, Xa, Ya, dev, epochs=3, lr=1e-3, bs=64)
    print("\n[D1] FULL-PARAM Waterbirds — per-layer gradient split")
    print("   ", probe_d1_perlayer(model, Xt, Yt, Gt, Xa, Ya))


if __name__ == "__main__":
    main()
