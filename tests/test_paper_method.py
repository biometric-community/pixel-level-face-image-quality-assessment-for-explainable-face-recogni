"""Model-specific quality (Eq. 1) and pixel-level quality maps (Eqs. 4-7), on a toy network."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pixfiq.eval import best_acc_eer
from pixfiq.quality.pixel_fiq import apply_blur, apply_center_occlusion, pixel_quality_map, scalar_quality


class ToyFR(nn.Module):
    """A deterministic linear face model: its embedding never varies under dropout."""

    def __init__(self, size: int = 8, dim: int = 6, classes: int = 4):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3 * size * size, dim)
        self.classifier = nn.Linear(dim, classes, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(classes, dim) * -1.0)
            self.classifier.weight[:, -1] = 0.0

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x.flatten(1)), dim=1)


def test_a_representation_that_never_varies_has_full_quality():
    """Eq. (1): QI = 2 sigmoid(-2/m^2 sum d(x_i, x_j)), which is 1 when the stochastic
    embeddings do not vary, whatever the classifier weights are."""
    qi = scalar_quality(ToyFR().eval(), torch.rand(3, 3, 8, 8))
    assert torch.allclose(qi, torch.ones(3), atol=1e-4)


def test_pixel_maps_are_not_rescaled_per_image():
    """Sec. 3.3: gradients are not min-max scaled per image, so maps stay comparable and
    v(S) in Eq. (6) is monotone in the raw gradients. Scaling the input by 10 shrinks this
    model's gradients tenfold, which a comparable map has to show."""
    torch.manual_seed(1)
    x = torch.rand(1, 3, 8, 8)
    _, maps = pixel_quality_map(ToyFR().eval(), torch.cat([x, 10 * x]))
    peaks = maps.flatten(1).max(dim=1).values
    assert peaks[0] > peaks[1] + 0.01


def test_pixel_maps_have_one_value_per_pixel_in_the_unit_range():
    """Eqs. (5) and (7): the three colour channels are merged into one map in [0, 1]."""
    qi, maps = pixel_quality_map(ToyFR().eval(), torch.rand(2, 3, 8, 8))
    assert maps.shape == (2, 8, 8)
    assert ((maps >= 0) & (maps <= 1)).all()
    assert qi.shape == (2,)


def test_disturbances_change_the_image_where_expected():
    x = torch.rand(1, 3, 16, 16)
    occluded = apply_center_occlusion(x)
    assert torch.equal(occluded[..., :4, :], x[..., :4, :])
    assert (occluded[..., 8, 8] == 0).all()
    assert apply_blur(x).shape == x.shape
    assert apply_blur(x).std() < x.std()


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
