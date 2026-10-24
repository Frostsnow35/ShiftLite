"""NNCF 训练后量化（PTQ）：OpenVINO FP32 IR → INT8 IR。

对应方案 §3.3 的 C 组。只量化生成器，校准集用 A 域（马）图像若干张。

⚠️ 实测结论（本机，2026-10-23）：
CycleGAN 生成器里的 InstanceNorm（IR 中表现为 MVN 算子）**不能一起量化**。
默认配置下整网量化后输出直接崩溃：PSNR(b,c)=12.3 dB、SSIM=0.17，生成图糊成一片。
把 MVN 排除出量化范围后恢复到 PSNR≈29 dB、SSIM≈0.93。
因此本模块默认 skip_instance_norm=True；要复现"崩溃"现象可加 --keep-instance-norm。

用法：
    python -m core.quantize --xml artifacts/models/horse2zebra.xml \
        --calib-dir artifacts/calib --subset 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .utils import load_image, to_tensor


def collect_calibration(calib_dir: Path, limit: int) -> list[np.ndarray]:
    """按文件名排序取前 limit 张，转成 NCHW float32 [-1,1]。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in Path(calib_dir).rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"校准目录里没有图片：{calib_dir}")
    files = files[:limit]
    return [to_tensor(load_image(p)).astype(np.float32) for p in files]


def quantize(xml_path: Path, calib: list[np.ndarray], out_xml: Path,
             skip_instance_norm: bool = True,
             accuracy_control: bool = False) -> tuple[Path, dict]:
    import nncf
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(xml_path))
    dataset = nncf.Dataset(calib, lambda x: x)

    kwargs = {
        "subset_size": len(calib),
        "preset": nncf.QuantizationPreset.MIXED,
    }
    if skip_instance_norm:
        # MVN = InstanceNorm 在 OpenVINO IR 里的算子名
        kwargs["ignored_scope"] = nncf.IgnoredScope(types=["MVN"])

    if accuracy_control:
        # 方案备选链第 2 档：以部分加速收益换取质量。
        # 需要一个"参考输出"校验集；这里用校准集前若干张的 FP32 输出做参照。
        fp32 = core.compile_model(str(xml_path), "CPU",
                                  {"INFERENCE_PRECISION_HINT": "f32"})
        subset = calib[:max(1, min(8, len(calib)))]
        refs = [np.asarray(fp32(x)[0]) for x in subset]

        def validation_fn(compiled, _ds):
            errs = [float(np.mean((np.asarray(compiled(x)[0]) - r) ** 2))
                    for x, r in zip(subset, refs)]
            return -float(np.mean(errs))  # 越大越好

        pairs = nncf.Dataset(list(zip(subset, refs)), lambda p: p[0])
        # MVN 保护对该分支同样适用：不传 ignored_scope 会把 InstanceNorm
        # 一起量化，结果与默认 PTQ 那次实测到的崩溃（12.3 dB）相同。
        ignored = kwargs.get("ignored_scope")
        quantized = nncf.quantize_with_accuracy_control(
            model, dataset, pairs, validation_fn,
            subset_size=len(calib),
            **({"ignored_scope": ignored} if ignored is not None else {}),
        )
        mode = "accuracy_control" + ("_skip_mvn" if skip_instance_norm else "")
    else:
        quantized = nncf.quantize(model, dataset, **kwargs)
        mode = "ptq_mixed_skip_mvn" if skip_instance_norm else "ptq_mixed"

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(out_xml), compress_to_fp16=False)

    info = {
        "mode": mode,
        "calibration_images": len(calib),
        "skip_instance_norm": skip_instance_norm,
        "quantized_xml": str(out_xml),
    }
    return out_xml, info


def main():
    ap = argparse.ArgumentParser(description="NNCF PTQ 量化")
    ap.add_argument("--xml", default="artifacts/models/horse2zebra.xml")
    ap.add_argument("--calib-dir", default="artifacts/calib")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--accuracy-control", action="store_true",
                    help="启用精度控制量化（用部分加速换质量）")
    ap.add_argument("--keep-instance-norm", action="store_true",
                    help="不排除 InstanceNorm；用于复现量化崩溃现象")
    ap.add_argument("--out", default="artifacts/models/horse2zebra_int8.xml")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise SystemExit(f"找不到 IR 模型：{xml_path}（先跑 cli.py setup）")

    print(f"[1/2] 收集校准数据（上限 {args.subset} 张）...", flush=True)
    calib = collect_calibration(Path(args.calib_dir), args.subset)
    print(f"      实际 {len(calib)} 张", flush=True)

    print("[2/2] NNCF 量化 ...", flush=True)
    out, info = quantize(xml_path, calib, Path(args.out),
                         skip_instance_norm=not args.keep_instance_norm,
                         accuracy_control=args.accuracy_control)
    size = out.stat().st_size + out.with_suffix(".bin").stat().st_size
    print(f"      → {out} ({size/1024/1024:.1f} MB)")

    (out.parent / "quantize_report.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
