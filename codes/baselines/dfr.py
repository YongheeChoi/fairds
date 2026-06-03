"""DFR (Deep Feature Reweighting, Kirichenko et al. 2022) baseline.

The standard last-layer-retraining method for spurious robustness: freeze the
backbone and retrain only the head on GROUP-BALANCED data. We implement the
balancing via inverse-frequency group weights on the per-sample loss. This is
the natural baseline for Fairds's last-layer Shapley variant — if Fairds only
matches DFR, we report that honestly (Codex Q2).

Note: uses group labels (oracle), like GroupDRO. The caller is responsible for
freezing the backbone (e.g. build_model(freeze_backbone=True)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairds.trainer import TrainLog, _accuracy


def train_dfr(model, loader, x_val, y_val, train_groups, n_groups=2,
              epochs=20, lr=0.1, device="cpu"):
    model = model.to(device)
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    counts = torch.bincount(train_groups.detach().cpu().long(), minlength=n_groups).float()
    group_w = (counts.sum() / (n_groups * counts.clamp_min(1.0))).to(device)  # inverse-freq

    log = TrainLog(method="dfr", epochs=epochs)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            x, y, _idx, grp = batch
            x, y, grp = x.to(device), y.to(device), grp.to(device)
            optim.zero_grad()
            per_sample = F.cross_entropy(model(x), y, reduction="none")
            loss = (group_w[grp] * per_sample).mean()
            loss.backward()
            optim.step()
        log.val_acc.append(_accuracy(model, x_val, y_val))
    return log
