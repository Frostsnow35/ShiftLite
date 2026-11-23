"""CycleGAN 小规模训练复现（horse2zebra，独立环节，对应 issue #8）。

硬边界（开题三节）：
1. 自训权重**不进** A/B/C 三组对照——训练引入超参/轮数/划分等新变量，
   会破坏「同源权重、只变运行环境与数值精度」的干净对照；
2. 训练结论**单独成表**，不与主对照共用行号或模型文件名。

用法（Colab 免费 T4 或本机 CPU 冒烟）：
    python train.py --data-root /path/to/horse2zebra --out runs/exp1
    python train.py --data-root ... --out runs/smoke --n-epochs 1 --max-iters 20  # 冒烟
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------- 模型（CycleGAN 论文结构：G=7 resblock，D=70x70 PatchGAN） ----------------


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


def build_generator(dim: int = 64, n_blocks: int = 7) -> nn.Module:
    layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, dim, 7), nn.InstanceNorm2d(dim),
              nn.ReLU(inplace=True)]
    for i in range(2):  # 下采样
        mult = 2 ** i
        layers += [nn.Conv2d(dim * mult, dim * mult * 2, 3, stride=2, padding=1),
                   nn.InstanceNorm2d(dim * mult * 2), nn.ReLU(inplace=True)]
    mult = 2 ** 2
    layers += [ResnetBlock(dim * mult) for _ in range(n_blocks)]
    for i in range(2):  # 上采样
        mult = 2 ** (2 - i)
        layers += [nn.ConvTranspose2d(dim * mult, dim * mult // 2, 3, stride=2,
                                      padding=1, output_padding=1),
                   nn.InstanceNorm2d(dim * mult // 2), nn.ReLU(inplace=True)]
    layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, 3, 7), nn.Tanh()]
    return nn.Sequential(*layers)


def build_discriminator(dim: int = 64) -> nn.Module:
    layers = [nn.Conv2d(3, dim, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
    for i in range(1, 3):
        mult = min(2 ** i, 8)
        layers += [nn.Conv2d(dim * mult // 2, dim * mult, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(dim * mult), nn.LeakyReLU(0.2, inplace=True)]
    layers += [nn.Conv2d(dim * 4, dim * 8, 4, stride=1, padding=1),
               nn.InstanceNorm2d(dim * 8), nn.LeakyReLU(0.2, inplace=True),
               nn.Conv2d(dim * 8, 1, 4, stride=1, padding=1)]
    return nn.Sequential(*layers)


# ---------------- 数据 ----------------


class UnpairedImages(Dataset):
    """两个域的无配对图像；训练时各自随机取样。"""

    def __init__(self, root_a: Path, root_b: Path, size: int = 256):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.a = sorted(p for p in root_a.rglob("*") if p.suffix.lower() in exts)
        self.b = sorted(p for p in root_b.rglob("*") if p.suffix.lower() in exts)
        if not self.a or not self.b:
            raise SystemExit(f"数据目录为空：{root_a} / {root_b}")
        self.tf = transforms.Compose([
            transforms.Resize(size), transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return max(len(self.a), len(self.b))

    def __getitem__(self, i):
        img_a = Image.open(self.a[i % len(self.a)]).convert("RGB")
        img_b = Image.open(self.b[random.randrange(len(self.b))]).convert("RGB")
        return self.tf(img_a), self.tf(img_b)


# ---------------- 训练 ----------------


class ImagePool:
    """历史生成图缓冲（论文默认 50），稳定判别器训练。"""

    def __init__(self, size: int = 50):
        self.size, self.images = size, []

    def query(self, img: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return img
        out = []
        for x in img.detach():
            x = x.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(x)
                out.append(x)
            elif random.random() > 0.5:
                idx = random.randrange(self.size)
                out.append(self.images[idx].clone())
                self.images[idx] = x
            else:
                out.append(x)
        return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description="CycleGAN 训练复现（不进 A/B/C 对照）")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="解压后的 horse2zebra/ 目录（含 trainA/trainB/testA/testB）")
    ap.add_argument("--out", type=Path, default=Path("runs/exp"))
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    ap.add_argument("--n-epochs", type=int, help="覆盖 config（恒定 lr 期）")
    ap.add_argument("--n-epochs-decay", type=int, help="覆盖 config（线性衰减期）")
    ap.add_argument("--max-iters", type=int, default=0, help=">0 时提前停（冒烟用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for k in ("n_epochs", "n_epochs_decay"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config_used.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    dev = torch.device(args.device)
    print(f"device={dev} config={cfg}", flush=True)

    ds = UnpairedImages(args.data_root / "trainA", args.data_root / "trainB",
                        cfg["image_size"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=0, drop_last=True)

    G_AB, G_BA = build_generator().to(dev), build_generator().to(dev)  # 马→斑马 / 斑马→马
    D_A, D_B = build_discriminator().to(dev), build_discriminator().to(dev)

    gan = nn.MSELoss()            # LSGAN
    cyc = nn.L1Loss()
    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                             lr=cfg["lr"], betas=(cfg["beta1"], 0.999))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                             lr=cfg["lr"], betas=(cfg["beta1"], 0.999))
    pool_fake_a, pool_fake_b = ImagePool(cfg["pool_size"]), ImagePool(cfg["pool_size"])

    def lr_at(epoch: int) -> float:
        """前 n_epochs 恒定，之后线性衰减到 0（论文 100+100）。"""
        if epoch <= cfg["n_epochs"]:
            return cfg["lr"]
        t = (epoch - cfg["n_epochs"]) / max(1, cfg["n_epochs_decay"])
        return cfg["lr"] * max(0.0, 1.0 - t)

    log_path = args.out / "train_losses.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(
            ["epoch", "iter", "g_ab_gan", "g_ba_gan", "cyc_a", "cyc_b",
             "idt_a", "idt_b", "d_a", "d_b", "lr"])

    it = 0
    t0 = time.time()
    total_epochs = cfg["n_epochs"] + cfg["n_epochs_decay"]
    for epoch in range(1, total_epochs + 1):
        lr = lr_at(epoch)
        for g in (opt_G, opt_D):
            for param_group in g.param_groups:
                param_group["lr"] = lr
        ep = dict(g_ab=0., g_ba=0., ca=0., cb=0., ia=0., ib=0., da=0., db=0., n=0)

        for real_a, real_b in dl:
            real_a, real_b = real_a.to(dev), real_b.to(dev)

            # ---- 生成器：GAN + 循环一致 + 身份 ----
            opt_G.zero_grad()
            fake_b, fake_a = G_AB(real_a), G_BA(real_b)
            rec_a, rec_b = G_BA(fake_b), G_AB(fake_a)
            loss_cyc = cyc(rec_a, real_a) * cfg["lambda_cycle"] \
                + cyc(rec_b, real_b) * cfg["lambda_cycle"]
            idt_a, idt_b = G_BA(real_a), G_AB(real_b)
            loss_idt = (cyc(idt_a, real_a) + cyc(idt_b, real_b)) \
                * cfg["lambda_cycle"] * cfg["lambda_identity"]
            out_db, out_da = D_B(fake_b), D_A(fake_a)
            loss_g_ab = gan(out_db, torch.ones_like(out_db))
            loss_g_ba = gan(out_da, torch.ones_like(out_da))
            loss_G = loss_g_ab + loss_g_ba + loss_cyc + loss_idt
            loss_G.backward()
            opt_G.step()

            # ---- 判别器（经历史池）；目标尺寸跟随 PatchGAN 输出（30×30），不是图像 ----
            opt_D.zero_grad()
            d_real_a, d_fake_a = D_A(real_a), D_A(pool_fake_a.query(fake_a))
            d_real_b, d_fake_b = D_B(real_b), D_B(pool_fake_b.query(fake_b))
            loss_da = 0.5 * (gan(d_real_a, torch.ones_like(d_real_a))
                             + gan(d_fake_a, torch.zeros_like(d_fake_a)))
            loss_db = 0.5 * (gan(d_real_b, torch.ones_like(d_real_b))
                             + gan(d_fake_b, torch.zeros_like(d_fake_b)))
            (loss_da + loss_db).backward()
            opt_D.step()

            ep["g_ab"] += loss_g_ab.item(); ep["g_ba"] += loss_g_ba.item()
            ep["ca"] += cyc(rec_a, real_a).item(); ep["cb"] += cyc(rec_b, real_b).item()
            ep["ia"] += cyc(idt_a, real_a).item(); ep["ib"] += cyc(idt_b, real_b).item()
            ep["da"] += loss_da.item(); ep["db"] += loss_db.item()
            ep["n"] += 1
            it += 1
            if args.max_iters and it >= args.max_iters:
                break

        n = max(1, ep["n"])
        row = [epoch, it] + [round(ep[k] / n, 5) for k in
                             ("g_ab", "g_ba", "ca", "cb", "ia", "ib", "da", "db")] + [lr]
        with log_path.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {epoch}/{total_epochs} it={it} "
              f"G_ab={row[2]:.3f} G_ba={row[3]:.3f} cyc=({row[4]:.3f},{row[5]:.3f}) "
              f"D=({row[8]:.3f},{row[9]:.3f}) lr={lr:.6f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

        if epoch % 10 == 0 or args.max_iters:
            ck = args.out / f"G_AB_ep{epoch}.pth"
            torch.save(G_AB.state_dict(), ck)
            print(f"  saved {ck}", flush=True)
        if args.max_iters and it >= args.max_iters:
            print("max-iters reached, smoke stop", flush=True)
            break

    torch.save(G_AB.state_dict(), args.out / "G_AB_final.pth")
    torch.save(G_BA.state_dict(), args.out / "G_BA_final.pth")
    print(f"done. losses={log_path}  elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
