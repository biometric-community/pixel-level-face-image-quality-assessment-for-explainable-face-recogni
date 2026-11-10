# DEVIATIONS — Pixel-Level FIQ

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train / eval corpora | MS1M / VGGFace2 / ICAO sets | CelebA + LFW/CALFW/CPLFW | Local public data |
| D2 | FR backbone | Paper FR nets | ResNet-18 CE | Lab twin stack |
| D3 | Quality node ω | SER-FIQ QI (m=100, p=0.5), node weights QI/‖e‖₁, v(S) with γ=7.5 | Same (Eqs. 1, 3–7); the optional Eq. 2 rescaling is not applied | Its α and r are tuned per network on Adience |
| D4 | ICAO incompliance suite | In-house | Synthetic blur/occlusion | Data unavailable |
