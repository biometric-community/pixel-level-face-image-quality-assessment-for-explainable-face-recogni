"""Model-specific scalar FIQ (SER-FIQ) + pixel-level quality maps (training-free)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stochastic_embeddings(model, images: torch.Tensor, passes: int) -> torch.Tensor:
    """(passes, B, D) embeddings with the model's dropout layers switched on."""
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    states = [m.training for m in dropouts]
    for m in dropouts:
        m.train()
    try:
        if hasattr(model, "trunk") and hasattr(model, "head"):
            features = model.trunk(images)  # deterministic part, computed once
            out = torch.stack([model.head(features) for _ in range(passes)])
        else:
            out = torch.stack([model.embed(images) for _ in range(passes)])
    finally:
        for m, state in zip(dropouts, states):
            m.train(state)
    return out


def scalar_quality(model, images: torch.Tensor, passes: int = 100) -> torch.Tensor:
    """
    Eq. 1 (SER-FIQ): QI = 2 sigmoid(-2/m^2 * sum_{i<j} d(x_i, x_j)) over m stochastic
    embeddings, so a representation that is robust to dropout has high quality.
    """
    with torch.no_grad():
        emb = _stochastic_embeddings(model, images, passes).transpose(0, 1)  # (B,m,D)
        dist = torch.cdist(emb, emb)  # (B,m,m)
        pair_sum = dist.sum(dim=(1, 2)) / 2.0  # sum over i < j
        return 2.0 * torch.sigmoid(-2.0 / passes**2 * pair_sum)


def visualise(saliency: torch.Tensor, gamma: float = 7.5) -> torch.Tensor:
    """Eq. 6: v(S) = 1 - 1 / (1 + 10^gamma * S^2), monotone in S and below 1."""
    return 1.0 - 1.0 / (1.0 + (10.0**gamma) * saliency**2)


def pixel_quality_map(
    model, images: torch.Tensor, passes: int = 100, gamma: float = 7.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eqs. 3-7: attach a quality node with weights QI / ||e_I||_1 to the embedding layer,
    back-propagate its output to the pixels, merge the channels by mean absolute gradient
    and map the result with v(). Maps are not rescaled per image, so they stay comparable.
    Returns (QI per image, pixel-level quality map in [0,1)).
    """
    model.eval()
    qi = scalar_quality(model, images, passes)
    x = images.detach().clone().requires_grad_(True)
    emb = model.embed(x)
    weights = (qi / emb.detach().abs().sum(dim=1)).unsqueeze(1)  # Eq. 3, fixed per image
    (weights * emb).sum().backward()  # Eq. 4: the quality node's output per image is QI
    saliency = x.grad.detach().abs().mean(dim=1)  # Eq. 5, (B,H,W)
    return qi.detach(), visualise(saliency, gamma)


def apply_blur(images: torch.Tensor, ksize: int = 9) -> torch.Tensor:
    pad = ksize // 2
    return F.avg_pool2d(images, kernel_size=ksize, stride=1, padding=pad)


def apply_center_occlusion(images: torch.Tensor, frac: float = 0.35) -> torch.Tensor:
    out = images.clone()
    _, _, h, w = out.shape
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    out[:, :, y0 : y0 + ch, x0 : x0 + cw] = 0.0
    return out


@torch.no_grad()
def disturbance_quality_drop(model, images: torch.Tensor) -> Dict[str, float]:
    model.eval()
    q_clean = scalar_quality(model, images).mean().item()
    q_blur = scalar_quality(model, apply_blur(images)).mean().item()
    q_occ = scalar_quality(model, apply_center_occlusion(images)).mean().item()
    return {
        "qi_clean": float(q_clean),
        "qi_blur": float(q_blur),
        "qi_occlusion": float(q_occ),
        "drop_blur": float(q_clean - q_blur),
        "drop_occlusion": float(q_clean - q_occ),
    }
