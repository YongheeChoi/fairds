"""EMA accumulator + softmax/clip weight transformer for Fairds."""

from __future__ import annotations

import torch
from torch import Tensor


class EMAReweighter:
    """Per-sample EMA of Shapley values, transformed into per-batch weights.

    The plan calls for: phi_i raw -> EMA accumulation (across iterations
    where sample i was visited) -> softmax/clip transformation -> w_i.

    For minibatch training, we maintain a buffer of per-sample EMA values
    indexed by global sample id. Samples that have never been seen start
    with EMA = 0 (uniform prior). At each minibatch step the new phi_i
    values are blended into the buffer with momentum.
    """

    def __init__(
        self,
        n_samples: int,
        momentum: float = 0.9,
        temperature: float = 1.0,
        clip_quantile: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_samples = n_samples
        self.momentum = momentum
        self.temperature = temperature
        self.clip_quantile = clip_quantile
        self.ema = torch.zeros(n_samples, device=device)
        self.seen = torch.zeros(n_samples, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, indices: Tensor, phi: Tensor) -> None:
        idx = indices.to(self.ema.device)
        new_vals = phi.to(self.ema.device).detach()
        old = self.ema[idx]
        unseen = ~self.seen[idx]
        # First-touch: set to phi directly to avoid zero-bias warmup.
        blended = torch.where(
            unseen, new_vals, self.momentum * old + (1.0 - self.momentum) * new_vals
        )
        self.ema[idx] = blended
        self.seen[idx] = True

    @torch.no_grad()
    def weights(self, indices: Tensor) -> Tensor:
        """Return per-sample weights for a minibatch, normalized to sum to len(indices)."""
        idx = indices.to(self.ema.device)
        vals = self.ema[idx].clone()

        if self.clip_quantile > 0:
            lo = torch.quantile(vals, self.clip_quantile)
            hi = torch.quantile(vals, 1.0 - self.clip_quantile)
            vals = vals.clamp(lo, hi)

        scaled = vals / max(self.temperature, 1e-8)
        # softmax for non-negativity then renormalize so mean weight == 1
        w = torch.softmax(scaled, dim=0)
        w = w * len(idx)
        return w
