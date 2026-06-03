"""In-Run Data Shapley closed-form Taylor approximations (Wang et al., 2024).

We treat the validation loss L_val as the utility function. For learning
rate eta and parameters theta_t, the Taylor expansion of utility change
when sample z_i is added to a gradient step is:

    Delta U_i ~ -eta * <g_i, g_val>                                      (1st-order)
              + (eta^2 / 2) * <g_i, H_val g_val> ...                     (2nd-order cross-term)

Following the plan, we use:
    phi_i^(1) = <g_i, g_val>
    phi_i^(2) = phi_i^(1) - alpha * <g_i, H_val g_val>

Sign conventions: a higher Shapley value means the sample reduces
validation loss faster, so a sample we want to upweight.

The 2nd-order cross-term measures the curvature-weighted alignment
between sample gradient and validation gradient. The plan's hypothesis
(C3, G5) is that the cross-term automatically penalizes majority-group
samples whose gradient is highly correlated with already-dominant
validation directions (representation bias attenuation).

All routines are written in pure PyTorch and operate on parameter
tensors directly so they are model-agnostic.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap


def _flatten_grads(grads: Iterable[Tensor]) -> Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def _per_sample_gradients(
    model: nn.Module,
    loss_fn,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    """Vectorized per-sample gradients via torch.func.vmap(grad(...)).

    Returns flattened tensor of shape (n, P) where P = total parameter count.
    Equivalent to running backward() once per sample but ~10–30x faster
    because the autograd graph is built once and reused.
    """

    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    param_keys = list(params.keys())

    def loss_one(params, x_i, y_i):
        out = functional_call(model, (params, buffers), (x_i.unsqueeze(0),))
        return loss_fn(out, y_i.unsqueeze(0))

    grad_fn = grad(loss_one, argnums=0)
    per_sample = vmap(grad_fn, in_dims=(None, 0, 0))(params, x, y)
    flat = torch.cat([per_sample[k].reshape(x.shape[0], -1) for k in param_keys], dim=1)
    return flat.detach()


def _validation_gradient(
    model: nn.Module,
    loss_fn,
    x_val: Tensor,
    y_val: Tensor,
) -> Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad = None
    loss = loss_fn(model(x_val), y_val)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return _flatten_grads(grads).detach()


def _hessian_vector_product(
    model: nn.Module,
    loss_fn,
    x_val: Tensor,
    y_val: Tensor,
    vec: Tensor,
) -> Tensor:
    """Compute H_val @ vec via Pearlmutter trick. Returns flat tensor."""

    params: List[Tensor] = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad = None

    loss = loss_fn(model(x_val), y_val)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = torch.cat([g.reshape(-1) for g in grads])
    inner = torch.dot(flat_grads, vec)
    hvp = torch.autograd.grad(inner, params, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in hvp]).detach()


def first_order_shapley_per_sample(
    model: nn.Module,
    loss_fn,
    x: Tensor,
    y: Tensor,
    x_val: Tensor,
    y_val: Tensor,
) -> Tensor:
    """Returns phi_i^(1) = <g_i, g_val> for each sample i, shape (n,).

    Higher value -> sample is more useful for validation loss reduction."""
    g_val = _validation_gradient(model, loss_fn, x_val, y_val)
    g_i = _per_sample_gradients(model, loss_fn, x, y)
    return g_i @ g_val  # (n,)


def second_order_shapley_per_sample(
    model: nn.Module,
    loss_fn,
    x: Tensor,
    y: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    alpha: float = 0.5,
    normalize_cross: bool = True,
) -> Tensor:
    """Returns phi_i^(2) = <g_i, g_val> - alpha * <g_i, H_val g_val>.

    The cross-term penalizes samples whose gradients align with directions
    that the validation Hessian already amplifies (typically, dominant /
    majority-group directions in representation space).

    With `normalize_cross=True` (default) the cross-term is rescaled so it
    has the same root-mean-square magnitude as the first-order term across
    the minibatch. This stops alpha from having to absorb a per-step
    scale that depends on the spectrum of H_val (which can vary by orders
    of magnitude as training progresses).
    """
    g_val = _validation_gradient(model, loss_fn, x_val, y_val)
    Hg_val = _hessian_vector_product(model, loss_fn, x_val, y_val, g_val)
    g_i = _per_sample_gradients(model, loss_fn, x, y)
    first = g_i @ g_val
    cross = g_i @ Hg_val
    if normalize_cross:
        eps = 1e-8
        rms_first = first.pow(2).mean().clamp_min(eps).sqrt()
        rms_cross = cross.pow(2).mean().clamp_min(eps).sqrt()
        cross = cross * (rms_first / rms_cross)
    return first - alpha * cross


def shapley_residual_arms(
    model: nn.Module,
    loss_fn,
    x: Tensor,
    y: Tensor,
    x_val: Tensor,
    y_val: Tensor,
):
    """Decompose the cross-term into its first-order-parallel part and its
    orthogonal residual, for the mechanism ablation (Codex Q1).

    Since g_val is ~the top eigenvector of H_val, cross_n is nearly parallel
    to phi1. We split:
        cross_n = beta * phi1 + r,   beta = <cross_n, phi1> / ||phi1||^2
    where r is the orthogonal residual (the curvature-specific signal that
    is NOT a rescaling of phi1). r is renormalized to phi1's RMS so a single
    gamma controls its magnitude fairly against the shuffled control.

    Returns (phi1, cross_n, r, beta).
    """
    eps = 1e-8
    g_val = _validation_gradient(model, loss_fn, x_val, y_val)
    Hg_val = _hessian_vector_product(model, loss_fn, x_val, y_val, g_val)
    g_i = _per_sample_gradients(model, loss_fn, x, y)
    phi1 = g_i @ g_val
    cross = g_i @ Hg_val
    rms_first = phi1.pow(2).mean().clamp_min(eps).sqrt()
    cross_n = cross * (rms_first / cross.pow(2).mean().clamp_min(eps).sqrt())
    beta = (cross_n @ phi1) / phi1.pow(2).sum().clamp_min(eps)
    r = cross_n - beta * phi1
    r = r * (rms_first / r.pow(2).mean().clamp_min(eps).sqrt())  # match phi1 RMS
    return phi1.detach(), cross_n.detach(), r.detach(), float(beta)
