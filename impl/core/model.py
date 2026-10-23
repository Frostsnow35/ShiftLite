"""CycleGAN 的 ResNet 生成器结构。

与官方仓库 junyanz/pytorch-CycleGAN-and-pix2pix 中 models/networks.py 的
ResnetGenerator 保持等价，确保能直接加载官方发布的 *.pth 权重。

horse2zebra 预训练模型用的是 --netG resnet_9blocks --norm instance：
    ngf = 64, n_blocks = 9, norm_layer = InstanceNorm2d(affine=False)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """残差块：卷积-归一化-激活-卷积-归一化，再与输入相加。"""

    def __init__(self, dim, padding_type="reflect", norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, use_bias=True):
        super().__init__()
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """CycleGAN 的生成器：下采样 2 次 → 9 个残差块 → 上采样 2 次。"""

    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 n_blocks=9, padding_type="reflect"):
        super().__init__()
        assert n_blocks >= 0
        use_bias = True

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2,
                          padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type,
                            norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def load_generator(weights_path: str, device: str = "cpu") -> ResnetGenerator:
    """加载官方 .pth 权重，返回 eval 模式的生成器。"""
    # OpenVINO 的设备名是 "CPU"，PyTorch 只认小写，这里统一归一化
    dev = "cpu" if str(device).lower() == "cpu" else device
    net = ResnetGenerator()
    state = torch.load(weights_path, map_location=dev, weights_only=False)

    # 官方权重可能直接是 state_dict，也可能包了一层 {"model": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"权重缺键 {len(missing)} 个，结构不匹配：{missing[:5]}")

    net.to(dev).eval()
    return net


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
