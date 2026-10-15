# FIDELITY_AUDIT — Pixel-Level FIQ

Five-pass checklist (paper-2-model).

| Pass | Claim | Status | Evidence |
|------|-------|--------|----------|
| 1 | Model-specific scalar FIQ | OK | `pixfiq/quality/pixel_fiq.py` `scalar_quality` (max cos to classifier W) |
| 2 | Pixel maps via QI backprop | OK | `pixel_quality_map`; eval `pixel_map_center_mean` |
| 3 | QI drops under incompliance | OK (synthetic) | blur/occlusion drops in `outputs/logs/full/eval.json` |
| 4 | FR utility measured | OK | LFW/CALFW/CPLFW verification |
| 5 | Documented deviations | OK | `DEVIATIONS.md` D1–D4 |

Full-protocol snapshot: LFW acc≈59.7%; QI clean≈0.997; blur/occ drop≈0.009.
