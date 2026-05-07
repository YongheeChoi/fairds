"""Learning to Reweight Examples for Robust Deep Learning (Ren et al., 2018).

This is FORML's direct prototype: 1-step bi-level meta-learning that
treats per-sample weights as variables to be optimized against a
small clean validation set on each minibatch.

Algorithm per step:
  (1) compute per-sample loss L_i(theta)
  (2) initialise epsilon_i = 0; loss_meta = sum_i (1/n + epsilon_i) * L_i
  (3) take a virtual gradient step: theta_hat = theta - lr * grad_theta loss_meta
  (4) compute val_loss(theta_hat)
  (5) eps_grad = grad_eps val_loss(theta_hat)
  (6) w_i = max(0, -eps_grad_i); normalize so sum(w) = batch_size (if 0, fall back uniform)
  (7) actual update: theta = theta - lr * sum_i w_i / B * grad_theta L_i

We do NOT add the FORML group-fairness exemplar layer here — that
requires sensitive-attribute labels. This Ren2018 implementation already
provides the "1-step bi-level reweighting" baseline (and is the fairness
upper-bound for any setup where the val set has the right group balance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class Ren2018Log:
    method: str = "ren2018"
    epochs: int = 0
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    weight_per_group: List[Dict[int, List[float]]] = field(default_factory=list)


def _accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def _loss(model, x, y):
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def train_ren2018(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: Tensor,
    y_val: Tensor,
    epochs: int = 20,
    lr: float = 0.1,
    device: torch.device | str = "cpu",
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> Ren2018Log:
    model = model.to(device)
    log = Ren2018Log(epochs=epochs)
    params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(epochs):
        model.train()
        epoch_w: Dict[int, List[float]] = {}
        for batch in train_loader:
            x, y, _idx, grp = batch
            x = x.to(device); y = y.to(device); grp = grp.to(device)
            B = x.size(0)

            # Step 1-3: virtual update with eps=0 starting point.
            # We use a manual 1-step update equivalent to Ren2018.
            # Per Ren et al. 2018, the meta-loss is:
            #     L_meta(theta; eps) = sum_i ((1/B) + eps_i) * L_i(theta)
            # so that at eps=0, the virtual step is a standard 1/B uniform SGD step.
            # Without the 1/B term, the virtual step at eps=0 is zero — bug fix
            # after Codex Round 2 review.
            eps = torch.zeros(B, device=device, requires_grad=True)
            out = model(x)
            per_sample = F.cross_entropy(out, y, reduction="none")
            uniform = torch.full_like(eps, 1.0 / B)
            meta_loss = ((uniform + eps) * per_sample).sum()

            grads = torch.autograd.grad(meta_loss, params, create_graph=True)
            updated = [p - lr * g for p, g in zip(params, grads)]

            # Step 4: val loss with the virtually-updated parameters.
            # We do this by temporarily replacing model parameters.
            # Easiest: functional pass via state_dict swap. To avoid that
            # complexity for arbitrary nn.Module we build a parameter-named
            # dict and use torch.func.functional_call.
            from torch.func import functional_call
            param_dict = dict(zip([n for n, _ in model.named_parameters() if _.requires_grad], updated))
            buffers = {n: b.detach() for n, b in model.named_buffers()}
            val_logits = functional_call(model, (param_dict, buffers), (x_val,))
            val_loss = F.cross_entropy(val_logits, y_val)

            # Step 5: gradient w.r.t. eps
            eps_grad = torch.autograd.grad(val_loss, eps)[0]

            # Step 6: w_i = max(0, -eps_grad_i), normalize
            w = (-eps_grad).clamp(min=0.0)
            w_sum = w.sum()
            if w_sum > 1e-8:
                w = w * (B / w_sum)
            else:
                w = torch.ones(B, device=device)
            w = w.detach()

            # Step 7: real weighted update
            for p in params:
                if p.grad is not None:
                    p.grad = None
            out = model(x)
            per_sample = F.cross_entropy(out, y, reduction="none")
            loss = (w * per_sample).mean()
            loss.backward()
            with torch.no_grad():
                for p in params:
                    p.add_(p.grad, alpha=-lr)

            for g_id in grp.unique():
                mask = (grp == g_id).cpu()
                epoch_w.setdefault(int(g_id.item()), []).extend(w[mask].detach().cpu().tolist())

        log.weight_per_group.append(epoch_w)
        if train_x is not None and train_y is not None:
            log.train_acc.append(_accuracy(model, train_x, train_y))
        log.val_acc.append(_accuracy(model, x_val, y_val))
        log.val_loss.append(_loss(model, x_val, y_val))
    return log
