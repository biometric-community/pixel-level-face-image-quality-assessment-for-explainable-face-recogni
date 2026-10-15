# IMPLEMENTATION_PLAN — Pixel-Level FIQ

## Paper

Training-free model-specific scalar FIQ + pixel quality maps via quality-node backprop.

## Lab twin

| Component | Choice |
|-----------|--------|
| Backbone | ResNet-18 CE on CelebA |
| Scalar QI | Peak softmax confidence |
| Pixel map | Grad of mean |emb| w.r.t. input |
| Validation | QI drop under blur / center occlusion |
| Utility | LFW/CALFW/CPLFW verification |
