# DEVIATIONS — Pixel-Level FIQ

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train / eval corpora | MS1M / VGGFace2 / ICAO sets | CelebA + LFW/CALFW/CPLFW | Local public data |
| D2 | FR backbone | Paper FR nets | ResNet-18 CE | Lab twin stack |
| D3 | Quality node ω | Learned/attached per paper | Max cosine to FR classifier weights (+ input grad maps) | Training-free model-specific QI |
| D4 | ICAO incompliance suite | In-house | Synthetic blur/occlusion | Data unavailable |
