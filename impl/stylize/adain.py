"""AdaIN（任意参考图风格迁移）推理侧实现 —— issue #32 支线。

与主线的隔离约定：
  - 本目录只服务于「任意参考图风格化」这条支线，**不进 A/B/C 三组主线口径**；
  - 产物落 `artifacts/adain/` 与 `stylize/weights/`，**严禁写 `artifacts/results/`**；
  - 指标不与 horse2zebra 混表（AdaIN 语义不变，只换画风，与实例迁移不是同一任务）。

参考实现：naoto0804/pytorch-AdaIN（MIT）。结构逐层对齐官方 `net.py`：
  encoder = VGG19 前 4 个 stage（到 relu4_1，即 children[:31]）
  AdaIN   = 内容特征按通道对齐到风格特征的均值/方差
  decoder = 9 层卷积 + 3 次最近邻上采样（3.51 M 参数 / 13.4 MB fp32）
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
ENCODER_FILE = WEIGHTS / "vgg_normalised.pth"
DECODER_FILE = WEIGHTS / "decoder.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad():
    return nn.ReflectionPad2d((1, 1, 1, 1))


def build_vgg() -> nn.Sequential:
    """完整的 VGG19（到 relu5_4，53 个子层）。

    `vgg_normalised.pth` 是**完整 VGG** 的 state_dict，所以必须建全套才能 load，
    之后再切前 31 层作为 encoder（官方 test.py 就是这么做的）。
    """
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        _pad(), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),        # 3  relu1-1
        _pad(), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),       # 6  relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        _pad(), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),      # 10 relu2-1
        _pad(), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),     # 13 relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        _pad(), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),     # 17 relu3-1
        _pad(), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),     # 26 relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        _pad(), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),     # 30 relu4-1 ← encoder 到此为止
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),     # 39 relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),     # 52 relu5-4
    )


def build_decoder() -> nn.Sequential:
    """官方 decoder：9 层卷积，3 次 scale=2 最近邻上采样。"""
    up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
    return nn.Sequential(
        _pad(), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(), up(),
        _pad(), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(), up(),
        _pad(), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(), up(),
        _pad(), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        _pad(), nn.Conv2d(64, 3, (3, 3)),
    )


def adain(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """按通道把内容特征标准化后，用风格特征的均值/方差重标定。

    注意：这正是「实例归一化统计量」。转进 IR 后它会被拆成
    ReduceMean / Subtract / Sqrt / Divide 等原语，**没有单个 MVN 算子可排除**，
    所以主线那条 `--skip-mvn` 的经验不能直接照搬（详见 README）。
    """
    mu_c = content.mean((2, 3), keepdim=True)
    sd_c = content.std((2, 3), keepdim=True) + eps
    mu_s = style.mean((2, 3), keepdim=True)
    sd_s = style.std((2, 3), keepdim=True) + eps
    return (content - mu_c) / sd_c * sd_s + mu_s


class Stylizer(nn.Module):
    """content + style → 风格化输出。alpha=1 完全采用风格统计量，0 退化为原图。"""

    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, content: torch.Tensor, style: torch.Tensor,
                alpha=1.0) -> torch.Tensor:
        """alpha 既可以是 float（PyTorch 直接用），也可以是 0/1 维张量。

        做成张量输入的原因：转成 IR 之后 alpha 若被 trace 成常量，
        OpenVINO / INT8 后端就没法在推理时调强度了。
        """
        feats_c = self.encoder(content)
        feats_s = self.encoder(style)
        t = adain(feats_c, feats_s)
        if torch.is_tensor(alpha):
            a = alpha.reshape(-1, 1, 1, 1)
            t = a * t + (1.0 - a) * feats_c
        else:
            t = alpha * t + (1.0 - alpha) * feats_c
        return self.decoder(t)


def load_stylizer(encoder_path: Path = ENCODER_FILE,
                  decoder_path: Path = DECODER_FILE) -> Stylizer:
    missing = [p for p in (encoder_path, decoder_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 AdaIN 权重：" + "、".join(str(p) for p in missing) +
            "（运行 python stylize/run.py download 获取）")

    vgg = build_vgg()
    vgg.load_state_dict(torch.load(encoder_path, map_location="cpu",
                                   weights_only=True))
    encoder = nn.Sequential(*list(vgg.children())[:31])   # 到 relu4_1

    decoder = build_decoder()
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu",
                                       weights_only=True))

    net = Stylizer(encoder, decoder).eval()
    for p in net.parameters():        # 推理阶段全部冻结
        p.requires_grad_(False)
    return net


# ---------------- 预处理 / 后处理（与官方 test.py 一致） ----------------

def to_tensor(img, size: int | None = None) -> torch.Tensor:
    """PIL(RGB) → [1,3,H,W]。

    - size 给定时把**短边**缩放到 size；不给则保持原分辨率；
    - 宽高随后对齐到 8 的倍数：encoder 有 3 次 ceil_mode 池化、decoder 有 3 次 2×
      上采样，不对齐会在输出边缘多出几行/几列。
    """
    import numpy as np

    w, h = img.size
    if size:
        s = size / min(w, h)
        w, h = round(w * s), round(h * s)
    w = max(8, int(round(w / 8)) * 8)
    h = max(8, int(round(h / 8)) * 8)
    if img.size != (w, h):
        img = img.resize((w, h), 2)      # 2 = BILINEAR

    x = torch.from_numpy(
        np.asarray(img, dtype="float32") / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def to_image(t: torch.Tensor):
    """[1,3,H,W] → PIL(RGB)，反归一化并裁剪到 0–1。"""
    import numpy as np
    from PIL import Image
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (t.detach() * std + mean).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255.0 + 0.5).astype(np.uint8))


def resize_style_to(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """官方做法：风格图先缩放到内容图尺寸（AdaIN 只用通道统计量，尺寸一致省事）。"""
    return F.interpolate(style, size=content.shape[-2:], mode="bilinear",
                         align_corners=False)


__all__ = ["Stylizer", "adain", "build_decoder", "build_vgg", "load_stylizer",
           "resize_style_to", "to_image", "to_tensor"]
