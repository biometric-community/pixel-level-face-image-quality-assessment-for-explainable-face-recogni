"""LFW utility + FIQ disturbance drops + pixel-map statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pixfiq.data.celeba import CelebAIdDataset, resolve_celeba_root
from pixfiq.data.pairs import build_calfw, build_cplfw, build_lfw
from pixfiq.models.fr_net import FaceEmbedder
from pixfiq.quality.pixel_fiq import disturbance_quality_drop, pixel_quality_map, scalar_quality


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_pairs(model, ds, device, batch_size, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    emb_dim = int(blob.get("emb_dim", cfg["model"].get("emb_dim", 128)))
    model = FaceEmbedder(int(blob["num_classes"]), emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    results: Dict = {"verification": {}, "fiq": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        results["verification"][name] = eval_pairs(model, ds, device, bs, nw, name)

    # FIQ on CelebA val subset
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    val_ds = CelebAIdDataset(
        root,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=int(cfg["train"].get("seed", 42)),
        train=False,
    )
    max_fiq = int(cfg["eval"].get("max_fiq_images", 256))
    idxs = list(range(min(len(val_ds), max_fiq)))

    class _Sub(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base, self.indices = base, indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.base[self.indices[i]]

    loader = DataLoader(_Sub(val_ds, idxs), batch_size=min(bs, 32), shuffle=False, num_workers=nw)
    drops, map_means, map_centers = [], [], []
    for images, _ in tqdm(loader, desc="fiq", leave=False):
        images = images.to(device)
        drops.append(disturbance_quality_drop(model, images))
        qi, sal = pixel_quality_map(model, images)
        map_means.append(float(sal.mean().cpu()))
        # center vs border mass (explainability proxy); sal is (B,H,W)
        h, w = sal.shape[-2], sal.shape[-1]
        cy0, cy1 = h // 4, 3 * h // 4
        cx0, cx1 = w // 4, 3 * w // 4
        center = sal[:, cy0:cy1, cx0:cx1].mean().item()
        map_centers.append(center)

    # aggregate drops
    fiq = {
        "qi_clean": float(np.mean([d["qi_clean"] for d in drops])),
        "qi_blur": float(np.mean([d["qi_blur"] for d in drops])),
        "qi_occlusion": float(np.mean([d["qi_occlusion"] for d in drops])),
        "drop_blur": float(np.mean([d["drop_blur"] for d in drops])),
        "drop_occlusion": float(np.mean([d["drop_occlusion"] for d in drops])),
        "pixel_map_mean": float(np.mean(map_means)),
        "pixel_map_center_mean": float(np.mean(map_centers)),
        "n_images": int(len(idxs)),
    }
    print(
        f"[fiq] clean={fiq['qi_clean']:.4f} blur_drop={fiq['drop_blur']:.4f} "
        f"occ_drop={fiq['drop_occlusion']:.4f} map_center={fiq['pixel_map_center_mean']:.4f}"
    )
    results["fiq"] = fiq

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
