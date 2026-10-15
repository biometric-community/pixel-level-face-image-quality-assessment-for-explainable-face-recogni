# REPORT — Pixel-Level Face Image Quality Assessment

**Protocol:** `full`

Lab twin of Terhorst et al.: model-specific scalar FIQ + gradient pixel maps
on a CelebA-trained ResNet-18; verified under blur/occlusion disturbances.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Elapsed (s): 1126.6

![Training](outputs/figures/full/train_loss.svg)

## Verification

| Dataset | Accuracy | EER | N |
|---------|----------|-----|---|
| calfw | 53.05% | 46.95% | 6000 |
| cplfw | 53.50% | 46.50% | 6000 |
| lfw | 59.70% | 40.30% | 6000 |

![Verification](outputs/figures/full/verification.svg)

## FIQ / explainability

- QI clean: **0.9967**
- Drop under blur: **0.0087**
- Drop under center occlusion: **0.0086**
- Pixel-map center mean: **0.1110** (n=512)

![FIQ](outputs/figures/full/fiq_disturbance.svg)

## Notes

- Paper uses MS1M/VGGFace2 + ICAO incompliance sets; this rebuild uses CelebA + LFW family.
- See `DEVIATIONS.md`.
