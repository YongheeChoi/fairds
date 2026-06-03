"""Minimal training loops: Vanilla SGD baseline and Fairds-weighted SGD.

Designed for E1's logistic regression scale. For larger models, replace
the per-sample gradient routine with the ghost-technique optimized
implementation (see Wang et al. 2024 reference repo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .reweighter import EMAReweighter
from .shapley import (
    first_order_shapley_per_sample,
    second_order_shapley_per_sample,
    shapley_residual_arms,
)


@dataclass
class TrainLog:
    method: str
    epochs: int
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    # phi by group (list of dicts: epoch -> {group_id: list of phi values})
    phi_per_group: List[Dict[int, List[float]]] = field(default_factory=list)
    weight_per_group: List[Dict[int, List[float]]] = field(default_factory=list)
    final_per_sample_phi_ema: Optional[List[float]] = None
    final_per_sample_weight: Optional[List[float]] = None
    # Diagnostic: per-batch magnitude tracking (added Round 1 fix).
    # Each list element is one minibatch.
    phi_std_per_batch: List[float] = field(default_factory=list)
    weight_std_per_batch: List[float] = field(default_factory=list)
    weight_max_per_batch: List[float] = field(default_factory=list)
    weight_min_per_batch: List[float] = field(default_factory=list)


def _accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=-1)
        return (pred == y).float().mean().item()


def _loss(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        out = model(x)
        return F.cross_entropy(out, y).item()


def _shuffle_residual_within(r, phi1, y, n_bins=4):
    """Permutation control: shuffle the residual within (class x phi1-quantile)
    bins, destroying any sample-specific residual signal while preserving its
    marginal distribution. If H* is real, real residual > shuffled residual."""
    r_s = r.clone()
    edges = torch.quantile(phi1, torch.linspace(0, 1, n_bins + 1, device=phi1.device)[1:-1])
    bins = torch.bucketize(phi1, edges)
    for c in y.unique():
        for b in range(n_bins):
            idx = ((y == c) & (bins == b)).nonzero(as_tuple=True)[0]
            if idx.numel() > 1:
                r_s[idx] = r[idx[torch.randperm(idx.numel(), device=r.device)]]
    return r_s


def _residual_arm_phi(model, loss_fn, x, y, x_val, y_val, alpha, arm):
    """Per-sample phi for the residual-ablation arms (Codex Q1)."""
    phi1, _cross_n, r, beta = shapley_residual_arms(model, loss_fn, x, y, x_val, y_val)
    if arm == "phi1":
        return phi1
    if arm == "parallel":  # (1 - alpha*beta) phi1 = phi1 rescaled = smoothing-equivalent
        return (1.0 - alpha * beta) * phi1
    if arm == "residual_real":
        return phi1 - alpha * r
    if arm == "residual_shuffle":
        return phi1 - alpha * _shuffle_residual_within(r, phi1, y)
    if arm == "sign_flip":
        return phi1 + alpha * r
    raise ValueError(f"unknown arm {arm}")


def train_vanilla(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: Tensor,
    y_val: Tensor,
    epochs: int = 20,
    lr: float = 0.1,
    device: torch.device | str = "cpu",
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
    train_groups: Optional[Tensor] = None,
) -> TrainLog:
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    log = TrainLog(method="vanilla", epochs=epochs)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, y, _idx, _grp = batch
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optim.step()

        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))
    return log


def train_fairds(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: Tensor,
    y_val: Tensor,
    n_train: int,
    train_groups: Tensor,
    order: int = 1,
    alpha: float = 0.5,
    momentum: float = 0.9,
    temperature: float = 0.5,
    clip_quantile: float = 0.05,
    epochs: int = 20,
    lr: float = 0.1,
    device: torch.device | str = "cpu",
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
    weight_scale: float = 1.0,
    warmup_epochs: int = 0,
    no_ema: bool = False,
    arm: Optional[str] = None,
) -> TrainLog:
    """
    weight_scale: post-softmax temperature-style amplifier; w' = (w-1)*scale + 1.
    Used to push reweighting magnitude beyond softmax saturation. scale=1 = vanilla
    softmax. scale=2 doubles deviation from uniform weights, etc.

    warmup_epochs: number of leading epochs to run with uniform weights (vanilla-style)
    before fairds reweighting kicks in. Useful when initialising from a pretrained
    model so the EMA buffer accumulates meaningful phi statistics before the
    weighting starts influencing the loss.

    no_ema: if True, skip the EMA buffer entirely and weight per-batch with the
    fresh phi values from the current step (Ren2018-style instantaneous, no
    cross-step accumulation noise). Only the per-batch softmax+clip+scale path
    runs. Useful in fine-tuning regimes where EMA accumulates noise from
    early-epoch small/noisy gradients.
    """
    """Weighted SGD with per-iteration in-run Shapley reweighting.

    `train_loader` must yield (x, y, sample_idx, group_id) tuples so we
    can index the EMA buffer and compute per-group statistics.
    """

    assert order in (1, 2)
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    reweighter = EMAReweighter(
        n_samples=n_train,
        momentum=momentum,
        temperature=temperature,
        clip_quantile=clip_quantile,
        device=device,
    )

    log = TrainLog(method=f"fairds-{order}", epochs=epochs)
    loss_fn = F.cross_entropy

    for epoch in range(epochs):
        model.train()
        epoch_phi: Dict[int, List[float]] = {}
        epoch_w: Dict[int, List[float]] = {}

        for batch in train_loader:
            x, y, idx, grp = batch
            x = x.to(device)
            y = y.to(device)
            grp = grp.to(device)

            # 1. Compute per-sample Shapley value at current parameters.
            if order == 1:
                phi = first_order_shapley_per_sample(model, loss_fn, x, y, x_val, y_val)
            elif arm is not None:
                phi = _residual_arm_phi(model, loss_fn, x, y, x_val, y_val, alpha, arm)
            else:
                phi = second_order_shapley_per_sample(
                    model, loss_fn, x, y, x_val, y_val, alpha=alpha
                )

            # 2. Update EMA buffer + read transformed weights.
            if no_ema:
                # Per-step weighting: softmax(phi/τ) on the fresh batch, no EMA.
                if epoch < warmup_epochs:
                    w = torch.ones(len(idx), device=device)
                else:
                    scaled = phi.detach() / max(temperature, 1e-8)
                    w = torch.softmax(scaled, dim=0) * len(idx)
                    if weight_scale != 1.0:
                        w = (w - 1.0) * weight_scale + 1.0
                        w = w.clamp_min(0.0)
                        w = w * (len(w) / w.sum().clamp_min(1e-8))
            else:
                reweighter.update(idx, phi)
                if epoch < warmup_epochs:
                    # During warmup, accumulate phi statistics but use uniform weights
                    # so the model fine-tunes normally first.
                    w = torch.ones(len(idx), device=device)
                else:
                    w = reweighter.weights(idx)
                    if weight_scale != 1.0:
                        w = (w - 1.0) * weight_scale + 1.0
                        w = w.clamp_min(0.0)
                        w = w * (len(w) / w.sum().clamp_min(1e-8))

            # Diagnostic: track magnitudes (one entry per batch).
            log.phi_std_per_batch.append(float(phi.detach().std().item()))
            log.weight_std_per_batch.append(float(w.detach().std().item()))
            log.weight_max_per_batch.append(float(w.detach().max().item()))
            log.weight_min_per_batch.append(float(w.detach().min().item()))

            # 3. Weighted SGD step (re-evaluate model on the batch).
            optim.zero_grad()
            out = model(x)
            per_sample_loss = F.cross_entropy(out, y, reduction="none")
            loss = (w.detach() * per_sample_loss).mean()
            loss.backward()
            optim.step()

            # bookkeeping
            for g_id in grp.unique():
                mask = (grp == g_id).cpu()
                epoch_phi.setdefault(int(g_id.item()), []).extend(
                    phi[mask].detach().cpu().tolist()
                )
                epoch_w.setdefault(int(g_id.item()), []).extend(
                    w[mask].detach().cpu().tolist()
                )

        log.phi_per_group.append(epoch_phi)
        log.weight_per_group.append(epoch_w)
        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))

    log.final_per_sample_phi_ema = reweighter.ema.detach().cpu().tolist()
    # Final per-sample weight (computed over all samples at once).
    with torch.no_grad():
        all_idx = torch.arange(n_train, device=reweighter.ema.device)
        log.final_per_sample_weight = reweighter.weights(all_idx).detach().cpu().tolist()
    return log
