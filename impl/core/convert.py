"""模型转换链路：PyTorch → OpenVINO IR（直转，不经 ONNX）。

对应方案 §3.2 PoC 与 §3.3 的转换环节。

为什么没有 ONNX 这一步
----------------------
开题报告第四节与 PLAN §3.3 定的口径是「由 PyTorch 前端直接生成 IR，不产生
ONNX 等中间格式」。2026-10-24 实测确认 OpenVINO 的 PyTorch 前端可以直接消化
生成器的两个"非标准"算子：

    nn.ReflectionPad2d  -> Pad(mode=reflect)
    nn.InstanceNorm2d   -> MVN

实际 IR 为 228 个节点、10 类算子，无需中间格式，因此保留直转即可。

用法：
    python -m core.convert --weights artifacts/models/horse2zebra.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import ResnetGenerator, load_generator
from .utils import SIZE, load_image, to_tensor

# 报告 §4 的验收判据用的是 0-255 灰度级，所以这里保留一个换算系数。
# 生成器输出层是 Tanh，值域 [-1, 1]，映射回 8bit 需要 (x+1)/2*255，
# 两个相邻灰度级之间的浮点间距因此是 2/255 ≈ 0.00784。
_GRAY_LEVEL = 2.0 / 255.0


def build_ir(weights: Path, xml_path: Path, fp16: bool = False,
             size: int = SIZE) -> Path:
    """PyTorch 权重 → OpenVINO IR（.xml + .bin）。

    ``input`` 必须显式给完整形状：只传 ``example_input`` 时前端会产出动态
    shape 的图，后续 CPU 编译与 NNCF 量化都会受影响。
    """
    import openvino as ov

    net = load_generator(str(weights))
    shape = [1, 3, size, size]

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        example = torch.zeros(*shape)
        model = ov.convert_model(net, example_input=example, input=shape)
    ov.save_model(model, str(xml_path), compress_to_fp16=fp16)
    return xml_path


def describe_ir(xml_path: Path) -> dict:
    """统计 IR 的规模与算子构成，用于回填 issue #4 的证据。"""
    import openvino as ov

    model = ov.Core().read_model(str(xml_path))
    ops: dict[str, int] = {}
    for op in model.get_ops():
        ops[op.get_type_name()] = ops.get(op.get_type_name(), 0) + 1
    return {
        "total_ops": sum(ops.values()),
        "op_kinds": len(ops),
        "op_histogram": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
    }


def _test_input(size: int, image: Path | None = None):
    """准备一张验证输入。

    优先用真实图像：官方计划的判据是"同一张图分别用 PyTorch 和 OpenVINO 各跑
    一遍"，一直用 `rng.standard_normal` 造出来的输入值域约 ±3，远超真实输入的
    [-1, 1]，虽然不影响耗时测量，但作为逐像素比对的那张"图"并不成立。
    """
    if image and Path(image).exists():
        x = to_tensor(load_image(image, size)).astype(np.float32)
        return x, Path(image).name
    rng = np.random.default_rng(20261014)
    x = rng.standard_normal((1, 3, size, size)).astype(np.float32)
    return x, None


def verify_against_torch(weights: Path, xml_path: Path,
                         size: int = SIZE,
                         image: Path | None = None) -> dict:
    """转换正确性判据：同一输入下 PyTorch 与 OpenVINO 输出的逐像素差。

    这一项就是 LPIPS(a,b) 要回答的问题：转换本身有没有引入偏差。

    判据沿用方案 §3.2：换算到 0-255 灰度后，平均 ≤ 0.5 灰度级、最大 ≤ 5 灰度级。
    `image` 为空时退回固定随机输入，`report["input_image"]` 会记为 None，
    报告里要如实写明。
    """
    import openvino as ov

    net = load_generator(str(weights))
    x, img_name = _test_input(size, image)

    with torch.no_grad():
        y_torch = net(torch.from_numpy(x)).numpy()

    core = ov.Core()
    # 必须显式指定 f32：OpenVINO 在部分 CPU 上默认按 bfloat16 推理，
    # 会把"转换正确性"污染成 2% 量级的假误差（实测 i5-1340P）。
    compiled = core.compile_model(str(xml_path), "CPU",
                                  {"INFERENCE_PRECISION_HINT": "f32"})
    y_ov = compiled(x)[0]

    diff = np.abs(y_torch - y_ov)
    gray = diff / _GRAY_LEVEL
    report = {
        "route": "pytorch->ov_ir (direct, no ONNX)",
        "input_image": img_name,
        "max_abs": round(float(diff.max()), 6),
        "mean_abs": round(float(diff.mean()), 6),
        "p99_abs": round(float(np.percentile(diff, 99)), 6),
        "same_shape": y_torch.shape == y_ov.shape,
        # 灰度级口径：与 §3.2 验收阈值直接可比
        "max_gray": round(float(gray.max()), 4),
        "mean_gray": round(float(gray.mean()), 4),
        "threshold": {"mean_gray": 0.5, "max_gray": 5.0},
        "pass": bool(gray.mean() <= 0.5 and gray.max() <= 5.0),
    }
    report.update(describe_ir(xml_path))
    return report


def main():
    ap = argparse.ArgumentParser(description="PyTorch → OpenVINO IR（直转）")
    ap.add_argument("--weights", default="artifacts/models/horse2zebra.pth")
    ap.add_argument("--out-dir", default="artifacts/models")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--image", default=None,
                    help="用作转换判据输入的那张图；不传则用固定随机输入")
    args = ap.parse_args()

    weights = Path(args.weights)
    out = Path(args.out_dir)
    if not weights.exists():
        raise SystemExit(f"找不到权重：{weights}")

    print(f"[1/2] PyTorch 直转 OpenVINO IR (输入 {args.size}x{args.size}) ...")
    xml_path = build_ir(weights, out / "horse2zebra.xml", args.fp16, args.size)
    print(f"      → {xml_path}")

    print("[2/2] 转换正确性验证 ...")
    report = verify_against_torch(weights, xml_path, args.size,
                                  Path(args.image) if args.image else None)
    print(f"      逐像素最大差 {report['max_abs']}（{report['max_gray']} 灰度级）")
    print(f"      逐像素平均差 {report['mean_abs']}（{report['mean_gray']} 灰度级）")
    print(f"      算子节点 {report['total_ops']} 个 / {report['op_kinds']} 类；"
          f"判据{'通过' if report['pass'] else '未通过'}")

    (out / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
