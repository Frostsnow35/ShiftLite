"""E4 层间分工分析：FP32 与模拟 INT8 两侧的逐层表示漂移。

设计依据（开题报告 §三 研究内容第三条）
------------------------------------
层间测量在 PyTorch 域内完成，不去 OpenVINO IR 上取中间结果：IR 没有对应钩子接口，
抽中间张量要把网络拆段导出，INT8 模型还要先反量化，测量误差会混进比较。
量化一侧用 NNCF 在 PyTorch 上做训练后量化（插 FakeQuantize 结点模拟 INT8 舍入），
与 `core/quantize.py` 对 IR 的量化共用同一套校准数据、preset 与忽略名单，
两侧精度损失同源，这里的层间结论可以对应到部署侧的 C 组。

用法：
    python -m core.layers --weights artifacts/models/horse2zebra.pth \
        --images artifacts/samples/horse_01.png --out artifacts/results/layers.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import ResnetBlock, load_generator

__all__ = [
    "enumerate_probe_layers",
    "capture_features",
    "build_simulated_int8",
    "compare_features",
    "merge_rows",
    "layer_report",
]


# ---------------- 探针位置 ----------------

def enumerate_probe_layers(net: nn.Module) -> list[tuple[str, nn.Module]]:
    """按前向顺序给出可挂载钩子的层，(层名, 模块)。

    取三类位置：下采样末段的 ReLU、全部残差块、上采样末段的 ReLU、最后一层卷积。
    残差块是重点——它承载转换前后每一级的残差表示。
    """
    seq = net.model
    probes: list[tuple[str, nn.Module]] = []
    n_down = 0
    n_res = 0
    n_up = 0

    for idx, mod in enumerate(seq):
        prev = seq[idx - 1] if idx > 0 else None
        prev2 = seq[idx - 2] if idx > 1 else None

        if isinstance(mod, ResnetBlock):
            n_res += 1
            probes.append((f"resblock_{n_res:02d}", mod))

        elif isinstance(mod, nn.ReLU) and isinstance(prev, nn.InstanceNorm2d) \
                and isinstance(prev2, nn.Conv2d):
            n_down += 1
            probes.append((f"down_relu_{n_down}", mod))

        elif isinstance(mod, nn.ConvTranspose2d):
            follow = seq[idx + 2] if idx + 2 < len(seq) else None
            if isinstance(follow, nn.ReLU):
                n_up += 1
                probes.append((f"up_relu_{n_up}", follow))

    head = seq[-2] if len(seq) >= 2 else None
    if isinstance(head, nn.Conv2d):
        probes.append(("head_conv", head))

    probes.sort(key=lambda p: _index_of(seq, p[1]))
    return probes


def _index_of(seq: nn.Sequential, mod: nn.Module) -> int:
    for i, m in enumerate(seq):
        if m is mod:
            return i
    return 1 << 30


# ---------------- 前向钩子 ----------------

def capture_features(net: nn.Module, x: torch.Tensor,
                     layer_names: list[str] | None = None) -> dict:
    """跑一次前向，返回 {层名: NCHW float32 ndarray}。"""
    probes = enumerate_probe_layers(net)
    if layer_names:
        probes = [(n, m) for n, m in probes if n in layer_names]

    feats: dict[str, np.ndarray] = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            feats[name] = out.detach().float().cpu().numpy()
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in probes]
    try:
        with torch.no_grad():
            net(x)
    finally:
        for h in handles:
            h.remove()
    return feats


# ---------------- 量化侧（NNCF FakeQuantize 模拟） ----------------

def build_simulated_int8(weights: str | Path, calib: list[np.ndarray],
                         skip_instance_norm: bool = True,
                         preset: str = "mixed") -> nn.Module:
    """在 PyTorch 上插 FakeQuantize 结点，模拟 INT8 舍入，不改变网络拓扑。

    与 `core/quantize.py` 的部署侧配置对齐：那里在 IR 上显式排除 MVN，
    这里对应"不让 InstanceNorm 参与量化"。

    ⚠️ 实测（2026-10-24，NNCF 3.4.0）：**不要**为此传
    ``nncf.IgnoredScope(types=["InstanceNorm2d"])`` —— 3.x 的 ignored_scope
    会做存在性校验，"InstanceNorm2d" 不是 NNCF 图里的合法算子名，直接抛
    `ValidationError: Ignored nodes that matches types ['InstanceNorm2d'] were
    not found`。原因是不需要这么做：NNCF torch 后端默认只给卷积插量化器，
    实测全网的 FakeQuantize 钩子形如 `model/N/conv2d/0`，InstanceNorm 上一个
    也没有，本身就是"排除 IN"的行为；这也解释了为什么"不排除 InstanceNorm
    就崩溃"的现象只出现在 OpenVINO 侧（那里 InstanceNorm 塌缩成 MVN，而 OV
    后端默认会把 MVN 一起量化）。
    """
    import nncf

    net = load_generator(str(weights))
    tensors = [torch.from_numpy(np.ascontiguousarray(a)) for a in calib]
    dataset = nncf.Dataset(tensors, lambda t: t)

    try:
        preset_map = {
            "mixed": nncf.QuantizationPreset.MIXED,
            "performance": nncf.QuantizationPreset.PERFORMANCE,
        }
        preset_arg = preset_map[preset]
    except KeyError:
        raise ValueError(f"未知 preset：{preset}，可选 {list(preset_map)}")

    return nncf.quantize(net, dataset, subset_size=len(tensors),
                         preset=preset_arg)


# ---------------- 逐层比较 ----------------

def _stack(feats_a: dict, feats_b: dict) -> list[str]:
    common = [k for k in feats_a if k in feats_b]
    return common


def layer_features(feats_a: dict, feats_b: dict) -> list[dict]:
    """逐层指标：余弦相似度与相对误差。

    相对误差的分母必须用**整层尺度**，不能逐元素除：探针层里有大量 ReLU 输出
    （下采样末段、上采样末段），其中相当一部分激活恰好为 0，逐元素除以 |a| 会在
    这些位置炸到 1e7 量级，把真实漂移完全盖掉（实测过：down_relu 的 mean_rel_err
    一度显示为 1.28e7，实际余弦是 0.9999）。这里统一取该层激活的平均幅值做分母，
    另给出一个与分母无关的指标 rel_l2（相对 Frobenius 范数误差）交叉核对。
    """
    rows = []
    for name in _stack(feats_a, feats_b):
        a = np.asarray(feats_a[name], dtype=np.float64).reshape(-1)
        b = np.asarray(feats_b[name], dtype=np.float64).reshape(-1)

        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(a @ b / denom) if denom else 1.0

        abs_err = np.abs(a - b)
        norm_a = float(np.linalg.norm(a))
        scale = float(np.abs(a).mean()) or 1e-12  # 整层平均幅值，抗零元素
        rows.append({
            "layer": name,
            "shape": list(np.asarray(feats_a[name]).shape),
            "cosine": round(cos, 6),
            # 与归一化方式无关的稳健指标：相对 L2 误差
            "rel_l2": round(float(np.linalg.norm(a - b) / norm_a)
                            if norm_a else 0.0, 6),
            "mean_abs_err": round(float(abs_err.mean()), 8),
            "mean_rel_err": round(float(abs_err.mean() / scale), 6),
            "p99_rel_err": round(float(np.percentile(abs_err, 99) / scale), 6),
            "max_rel_err": round(float(abs_err.max() / scale), 6),
            "activation_mean_abs": round(scale, 6),
            "activation_max": round(float(np.abs(a).max()), 5),
        })
    return rows


def merge_rows(acc: dict[str, list[dict]]) -> list[dict]:
    """把逐图逐层的指标行按层聚合（取均值）。"""
    out = []
    for name, rows in acc.items():
        merged = {
            "layer": name,
            "shape": rows[0]["shape"],
            "n_images": len(rows),
        }
        for key in ("cosine", "rel_l2", "mean_abs_err", "mean_rel_err",
                    "p99_rel_err", "max_rel_err",
                    "activation_mean_abs", "activation_max"):
            merged[key] = round(float(np.mean([r[key] for r in rows])), 8)
        out.append(merged)
    out.sort(key=lambda r: r["layer"])
    return out


def compare_features(feats_list_a: list, feats_list_b: list) -> list[dict]:
    """多图结果按层聚合（每张图逐层算，最后取均值）。"""
    acc: dict[str, list[dict]] = {}
    for fa, fb in zip(feats_list_a, feats_list_b):
        for row in layer_features(fa, fb):
            acc.setdefault(row["layer"], []).append(row)
    return merge_rows(acc)


# ---------------- 主干流程 ----------------

def load_images(paths: list[Path], size: int = 256) -> list[np.ndarray]:
    from .utils import load_image, to_tensor
    return [to_tensor(load_image(p, size)) for p in paths]


def layer_report(weights: str | Path, calib: list[np.ndarray],
                 images: list[np.ndarray]) -> dict:
    """产出 E4 报告：两侧逐图前向、逐层对齐后聚合漂移指标。

    流式处理，逐图算完即释放特征：15 个探针层的全分辨率特征图每张两侧
    约 300 MB，120 张全部驻留会到 30 GB 级（实测段错误，2026-10-24，
    6 张样例时代无此问题）。
    """
    base = load_generator(str(weights))
    quant = build_simulated_int8(weights, calib)

    acc: dict[str, list[dict]] = {}
    for a in images:
        x = torch.from_numpy(np.ascontiguousarray(a))
        fa = capture_features(base, x)
        fb = capture_features(quant, x)
        for row in layer_features(fa, fb):
            acc.setdefault(row["layer"], []).append(row)
        del fa, fb, x

    rows = merge_rows(acc)
    probe_names = [n for n, _ in enumerate_probe_layers(base)]

    return {
        "n_images": len(images),
        "n_calibration": len(calib),
        "probe_layers": probe_names,
        "rows": rows,
        "summary": _summarize(rows),
    }


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    res = [r for r in rows if r["layer"].startswith("resblock")]
    worst = min(rows, key=lambda r: r["cosine"])
    best = max(rows, key=lambda r: r["cosine"])
    # 残差块天然按深度有序，看余弦是否单调下降（漂移沿网络深度累积的证据）
    res_seq = sorted(res, key=lambda r: r["layer"])
    cos_seq = [r["cosine"] for r in res_seq]
    l2_seq = [r["rel_l2"] for r in res_seq]
    monotone = (len(cos_seq) > 1
                and all(x >= y - 1e-9 for x, y in zip(cos_seq, cos_seq[1:])))
    return {
        "layers_compared": len(rows),
        "mean_cosine": round(float(np.mean([r["cosine"] for r in rows])), 6),
        "mean_rel_l2": round(float(np.mean([r["rel_l2"] for r in rows])), 6),
        "most_drifted_layer": worst["layer"],
        "most_drifted_cosine": worst["cosine"],
        "least_drifted_layer": best["layer"],
        "least_drifted_cosine": best["cosine"],
        "resblock_mean_rel_err": (
            round(float(np.mean([r["mean_rel_err"] for r in res])), 6)
            if res else None),
        # 深度方向的漂移趋势（E4 要回答的核心问题）
        "resblock_cosine_head_tail": (
            [cos_seq[0], cos_seq[-1]] if cos_seq else None),
        "resblock_rel_l2_head_tail": (
            [l2_seq[0], l2_seq[-1]] if l2_seq else None),
        "drift_monotonic_along_depth": bool(monotone),
    }


def main():
    import argparse

    from .quantize import collect_calibration

    ap = argparse.ArgumentParser(description="E4 层间分工分析")
    ap.add_argument("--weights", default="artifacts/models/horse2zebra.pth")
    ap.add_argument("--calib-dir", default="artifacts/calib")
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--images", nargs="+",
                    default=["artifacts/samples/horse_01.png"])
    ap.add_argument("--out", default="artifacts/results/layers.json")
    args = ap.parse_args()

    calib = collect_calibration(Path(args.calib_dir), args.subset)
    images = load_images([Path(p) for p in args.images])
    print(f"校准 {len(calib)} 张 / 评测图 {len(images)} 张")

    report = layer_report(args.weights, calib, images)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"\n{'层':<18}{'余弦':>10}{'相对L2':>10}{'平均相对误差':>14}")
    for r in report["rows"]:
        print(f"{r['layer']:<18}{r['cosine']:>10.4f}{r['rel_l2']:>10.4f}"
              f"{r['mean_rel_err']:>14.4f}")
    print(f"\n已写入 {out}")
    print(f"漂移最大层：{report['summary']['most_drifted_layer']} "
          f"({report['summary']['most_drifted_cosine']:.4f})")


if __name__ == "__main__":
    main()
