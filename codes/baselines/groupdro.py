"""GroupDRO (Sagawa et al. 2020).

Distributionally Robust Optimization over groups: at each step, compute
per-group loss, take the *worst-group* loss as the objective. Online
adaptive weights via exponential update on group losses.

This implementation requires per-sample group labels at training time
(stronger than JTT, weaker than methods that use sensitive labels in
the validation anchor — same regime as bi-level methods).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class GroupDROLog:
    method: str = "groupdro"
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


def train_groupdro(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: Tensor,
    y_val: Tensor,
    n_groups: int = 2,
    eta_q: float = 0.01,
    epochs: int = 20,
    lr: float = 0.05,
    device: torch.device | str = "cpu",
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> GroupDROLog:
    """Online GroupDRO with exponentiated-gradient adversary on group weights.

    Args:
        n_groups: number of distinct groups (assumes ids 0..n_groups-1).
        eta_q: step size for the adversary's exponentiated update.
    """
    model = model.to(device)
    log = GroupDROLog(epochs=epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=lr)
    q = torch.ones(n_groups, device=device) / n_groups  # adversary's group weights

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, y, _idx, g = batch
            x = x.to(device); y = y.to(device); g = g.to(device)

            optim.zero_grad()
            per = F.cross_entropy(model(x), y, reduction="none")
            # Per-group mean loss
            losses_g = torch.zeros(n_groups, device=device)
            counts_g = torch.zeros(n_groups, device=device)
            for gid in range(n_groups):
                m = (g == gid)
                if m.any():
                    losses_g[gid] = per[m].mean()
                    counts_g[gid] = m.sum().float()
            # Update adversary q on observed losses (exponentiated step)
            with torch.no_grad():
                q_new = q * torch.exp(eta_q * losses_g)
                # Only update where group was present in batch (counts_g > 0)
                q = torch.where(counts_g > 0, q_new, q)
                q = q / q.sum().clamp_min(1e-12)
            # Robust loss = sum_g q_g * losses_g (only over present groups)
            loss = (q.detach() * losses_g).sum()
            loss.backward()
            optim.step()

        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))

    return log
