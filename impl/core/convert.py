"""模型转换链路：PyTorch → ONNX → OpenVINO IR。

对应方案 §3.2 PoC 与 §3.3 小改进中的转换环节。

用法：
    python -m cyclelite.core.convert --weights artifacts/models/horse2zebra.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import ResnetGenerator, load_generator
from .utils import SIZE


def export_onnx(weights: Path, onnx_path: Path, opset: int = 13) -> Path:
    """导出 ONNX。生成器里的 ReflectionPad2d 走 Pad(reflect)，
    InstanceNorm 走 InstanceNormalization，opset 13 都能表达。"""
    net = load_generator(str(weights))
    dummy = torch.randn(1, 3, SIZE, SIZE)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    try:
        torch.onnx.export(net, dummy, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, dummy, str(onnx_path), **kwargs)
    return onnx_path


def to_openvino(onnx_path: Path, xml_path: Path, fp16: bool = False) -> Path:
    """ONNX → OpenVINO IR（.xml + .bin）。"""
    import openvino as ov

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path), input=[1, 3, SIZE, SIZE])
    ov.save_model(model, str(xml_path), compress_to_fp16=fp16)
    return xml_path


def verify_against_torch(weights: Path, xml_path: Path) -> dict:
    """转换正确性判据：同一输入下 PyTorch 与 OpenVINO 输出的逐像素差。

    这一项就是 LPIPS(a,b) 要回答的问题：转换本身有没有引入偏差。
    """
    import openvino as ov

    net = load_generator(str(weights))
    rng = np.random.default_rng(20261014)
    x = rng.standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)

    with torch.no_grad():
        y_torch = net(torch.from_numpy(x)).numpy()

    core = ov.Core()
    # 必须显式指定 f32：OpenVINO 在部分 CPU 上默认按 bfloat16 推理，
    # 会把"转换正确性"污染成 2% 量级的假误差（实测 i5-1340P）。
    compiled = core.compile_model(str(xml_path), "CPU",
                                  {"INFERENCE_PRECISION_HINT": "f32"})
    y_ov = compiled(x)[0]

    diff = np.abs(y_torch - y_ov)
    return {
        "max_abs": round(float(diff.max()), 6),
        "mean_abs": round(float(diff.mean()), 6),
        "p99_abs": round(float(np.percentile(diff, 99)), 6),
        "same_shape": y_torch.shape == y_ov.shape,
    }


def main():
    ap = argparse.ArgumentParser(description="PyTorch → ONNX → OpenVINO 转换")
    ap.add_argument("--weights", default="artifacts/models/horse2zebra.pth")
    ap.add_argument("--out-dir", default="artifacts/models")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    weights = Path(args.weights)
    out = Path(args.out_dir)
    if not weights.exists():
        raise SystemExit(f"找不到权重：{weights}")

    print(f"[1/3] 导出 ONNX (opset={args.opset}) ...")
    onnx_path = export_onnx(weights, out / "horse2zebra.onnx", args.opset)
    print(f"      → {onnx_path} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")

    print("[2/3] 转换 OpenVINO IR ...")
    xml_path = to_openvino(onnx_path, out / "horse2zebra.xml")
    print(f"      → {xml_path}")

    print("[3/3] 转换正确性验证 ...")
    report = verify_against_torch(weights, xml_path)
    print(f"      逐像素最大差 {report['max_abs']}，均值 {report['mean_abs']}")

    import json
    (out / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
