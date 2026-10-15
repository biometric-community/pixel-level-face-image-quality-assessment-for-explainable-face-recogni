# Pixel-Level Face Image Quality Assessment for Explainable Face Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of **pixel-level FIQ**: model-specific scalar quality + gradient pixel maps for explainable face recognition.

**Repository:** https://github.com/biometric-community/pixel-level-face-image-quality-assessment-for-explainable-face-recogni

**Paper:** [10.1109/TBIOM.2023.3263186](https://doi.org/10.1109/tbiom.2023.3263186)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
