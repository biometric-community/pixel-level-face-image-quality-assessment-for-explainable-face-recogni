"""Scalar QI + pixel quality map for one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pixfiq.models.fr_net import FaceEmbedder
from pixfiq.quality.pixel_fiq import pixel_quality_map, scalar_quality


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device(
        device_name if not str(device_name).startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    blob = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)
    model = FaceEmbedder(int(blob["num_classes"]), emb_dim=int(blob.get("emb_dim", 128)), pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        qi = float(scalar_quality(model, x)[0].cpu())
    _, sal = pixel_quality_map(model, x)
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fiq.json"
    with open(out, "w") as f:
        json.dump({"qi": qi, "pixel_map": sal[0].cpu().numpy().tolist()}, f)
        f.write("\n")
    print(f"Wrote {out} qi={qi:.4f}")


if __name__ == "__main__":
    main()
