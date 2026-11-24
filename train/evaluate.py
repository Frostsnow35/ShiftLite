"""训练过程评测：按 E3 口径算 FID/KID 曲线（生成批=testA 输出，真实批=testB）。

与主对照口径一致：不混用 trainA；FID/KID 复用 impl/core/metrics_deep（InceptionV3
2048 维 + 纯 numpy Frechet / 多项式核 MMD），与 E3 数字同源可比。

用法：
    python evaluate.py --data-root /path/to/horse2zebra --ckpt runs/exp/G_AB_ep10.pth \
        --curve runs/exp/fid_kid_curve.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "impl"))
from core.metrics_deep import inception_features, fid, kid  # noqa: E402
from train import build_generator  # noqa: E402


def load_images(paths: list[Path], size: int = 256) -> list[np.ndarray]:
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        out.append(np.asarray(im, dtype=np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser(description="FID/KID 曲线评测（E3 口径）")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="G_AB 权重（.pth）")
    ap.add_argument("--curve", type=Path, required=True, help="追加写入的 CSV")
    ap.add_argument("--limit", type=int, default=120, help="testA 抽样上限（正式口径 120）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_a = sorted(p for p in (args.data_root / "testA").rglob("*")
                    if p.suffix.lower() in exts)[:args.limit]
    test_b = sorted(p for p in (args.data_root / "testB").rglob("*")
                    if p.suffix.lower() in exts)
    if not test_a or not test_b:
        raise SystemExit("testA/testB 为空")

    G = build_generator().to(dev)
    G.load_state_dict(torch.load(args.ckpt, map_location=dev))
    G.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    fake_b = []
    with torch.no_grad():
        for p in test_a:
            x = tf(Image.open(p).convert("RGB").resize((256, 256))).unsqueeze(0).to(dev)
            y = G(x)[0].cpu()
            arr = ((y.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            fake_b.append(arr)
    real_b = load_images(test_b)

    f_fake = inception_features(fake_b)
    f_real = inception_features(real_b)
    fid_v, kid_v = fid(f_fake, f_real), kid(f_fake, f_real)
    print(f"FID={fid_v:.2f}  KID={kid_v:.4f}  (n_gen={len(fake_b)} n_real={len(real_b)})")

    args.curve.parent.mkdir(parents=True, exist_ok=True)
    new = not args.curve.exists()
    with args.curve.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ckpt", "n_gen", "n_real", "fid", "kid"])
        w.writerow([str(args.ckpt), len(fake_b), len(real_b),
                    round(fid_v, 4), round(kid_v, 6)])
    print(f"appended → {args.curve}")


if __name__ == "__main__":
    main()
