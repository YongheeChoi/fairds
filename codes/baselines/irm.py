"""IRM v1 (Arjovsky et al. 2019).

Loss = sum_e L_e(theta) + lambda * sum_e || grad_w (L_e(theta * w))_{w=1.0} ||^2

In binary classification with cross-entropy and a single dummy classifier
weight w=1.0 ("phantom classifier"), the IRM v1 penalty reduces to
||grad_w sum_y CE(model(x)*w, y)||^2 evaluated at w=1.0 per environment.

We adapt to the (group=environment) setup standard in spurious
benchmarks. The penalty discourages the classifier from being optimal
in *some* environment but not in another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class IRMLog:
    method: str = "irm"
    epochs: int = 0
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


def _accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def _loss(model, x, y):
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def _irm_penalty(logits: Tensor, y: Tensor) -> Tensor:
    """IRM v1 penalty: ||grad_w CE(logits*w, y)|_{w=1.0}||^2."""
    w = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = F.cross_entropy(logits * w, y)
    grad = torch.autograd.grad(loss, [w], create_graph=True)[0]
    return grad.pow(2)


def train_irm(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: Tensor,
    y_val: Tensor,
    n_groups: int = 2,
    irm_lambda: float = 100.0,
    irm_anneal_epochs: int = 5,
    epochs: int = 20,
    lr: float = 0.05,
    device: torch.device | str = "cpu",
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> IRMLog:
    """IRM v1 with linear penalty annealing.

    Args:
        n_groups: number of training environments.
        irm_lambda: full penalty weight (after annealing).
        irm_anneal_epochs: linear ramp from 1.0 to irm_lambda over this many epochs.
    """
    model = model.to(device)
    log = IRMLog(epochs=epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=lr)

    for epoch in range(epochs):
        model.train()
        if epoch < irm_anneal_epochs:
            lam = 1.0 + (irm_lambda - 1.0) * (epoch / max(1, irm_anneal_epochs - 1))
        else:
            lam = irm_lambda

        for batch in train_loader:
            x, y, _idx, g = batch
            x = x.to(device); y = y.to(device); g = g.to(device)
            logits = model(x)
            erm = F.cross_entropy(logits, y)
            penalty = torch.tensor(0.0, device=device)
            for gid in range(n_groups):
                m = (g == gid)
                if m.any():
                    penalty = penalty + _irm_penalty(logits[m], y[m])
            loss = erm + lam * penalty
            # IRM trick: rescale by lambda so optim doesn't see exploding loss
            if lam > 1.0:
                loss = loss / lam
            optim.zero_grad()
            loss.backward()
            optim.step()

        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))

    return log
