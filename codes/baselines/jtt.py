"""Just Train Twice (JTT, Liu et al. 2021).

Two-stage training:
  Stage 1 (ERM, T epochs): standard SGD; record which training samples
  the model misclassifies at the end of stage 1.
  Stage 2 (upweighted ERM, T epochs): run SGD again on the same data,
  but with weight `lambda_up` (>1) on stage-1-misclassified samples and 1
  on the rest.

JTT is a strong baseline for spurious-correlation benchmarks because
it captures the intuition that minority-group samples are more often
misclassified by ERM and should be upweighted, without needing group
labels at training time.

This implementation does not assume per-sample group labels; it only
needs the binary label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class JTTLog:
    method: str = "jtt"
    epochs: int = 0
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    n_misclassified_stage1: int = 0


def _accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def _loss(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def train_jtt(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: Tensor,
    y_val: Tensor,
    epochs: int = 20,
    lr: float = 0.05,
    lambda_up: float = 5.0,
    stage1_frac: float = 0.5,
    device: torch.device | str = "cpu",
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> JTTLog:
    """Train via JTT (just-train-twice).

    Args:
        epochs: total epochs (split between stages by `stage1_frac`).
        stage1_frac: fraction of epochs spent in stage 1 ERM.
        lambda_up: upweight factor for stage-1-misclassified samples in stage 2.
    """
    model = model.to(device)
    log = JTTLog(epochs=epochs)
    params = [p for p in model.parameters() if p.requires_grad]

    stage1_epochs = max(1, int(epochs * stage1_frac))
    stage2_epochs = epochs - stage1_epochs

    optim = torch.optim.SGD(params, lr=lr)

    # ---- Stage 1: vanilla ERM ----
    n_train = train_loader.dataset.__len__()
    for ep in range(stage1_epochs):
        model.train()
        for batch in train_loader:
            x, y, _idx, _grp = batch
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optim.step()
        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))

    # ---- End of stage 1: identify misclassified samples ----
    model.eval()
    miscls = torch.zeros(n_train, dtype=torch.bool, device=device)
    with torch.no_grad():
        for batch in train_loader:
            x, y, idx, _grp = batch
            x = x.to(device); y = y.to(device); idx = idx.to(device)
            pred = model(x).argmax(-1)
            miscls[idx] = pred != y
    log.n_misclassified_stage1 = int(miscls.sum().item())

    # ---- Stage 2: upweighted ERM ----
    optim = torch.optim.SGD(params, lr=lr)  # fresh optimizer for stage 2
    for ep in range(stage2_epochs):
        model.train()
        for batch in train_loader:
            x, y, idx, _grp = batch
            x = x.to(device); y = y.to(device); idx = idx.to(device)
            w = torch.where(miscls[idx], lambda_up, torch.ones_like(idx, dtype=torch.float32))
            optim.zero_grad()
            per = F.cross_entropy(model(x), y, reduction="none")
            loss = (w * per).mean()
            loss.backward()
            optim.step()
        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))

    return log
