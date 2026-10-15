"""Model-specific scalar FIQ + pixel-level quality maps (training-free)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def scalar_quality(model, images: torch.Tensor) -> torch.Tensor:
    """
    Model-specific QI: max cosine similarity of the embedding to classifier
    weight rows (how decisively the FR head recognizes the sample).
    """
    emb = model.embed(images)
    w = F.normalize(model.classifier.weight, p=2, dim=1)  # (C, D)
    # emb already L2-normalized → cosines = emb @ w.T
    return (emb @ w.t()).max(dim=1).values


def pixel_quality_map(model, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backprop model-specific QI → input pixels (paper-style quality-node grad).
    Returns (QI per image, |grad| channel-mean map in [0,1]).
    """
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    emb = model.embed(x)
    w = F.normalize(model.classifier.weight, p=2, dim=1)
    qi = (emb @ w.t()).max(dim=1).values
    qi.sum().backward()
    grad = x.grad.detach()
    sal = grad.abs().mean(dim=1)  # (B,H,W)
    b = sal.shape[0]
    flat = sal.view(b, -1)
    mn = flat.min(dim=1).values.view(b, 1, 1)
    mx = flat.max(dim=1).values.view(b, 1, 1)
    sal = (sal - mn) / (mx - mn + 1e-8)
    return qi.detach(), sal


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
