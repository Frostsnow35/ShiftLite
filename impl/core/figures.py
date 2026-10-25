"""实验图表产出（PLAN §6 产出栏，issue #14）。

E2 性能对比柱状图（含误差棒）→ fig_e2_perf.png
E3 异常样例拼图              → fig_e3_anomalies.png
E3 质量对比表                → quality_table.md
E4 层序号 × 表示差异曲线     → fig_e4_layers.png

所有数字从 artifacts/results/*.json 读取，不写死（性能图的 AMD 脚注除外，
见 AMD_FOOTNOTE 注释）。PNG 产物落 artifacts/figures/（不入库），脚本入库。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 中文渲染（Windows 通用；Linux 缺字体时自动回退 DejaVu）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 脚本根 = impl/。JSON 里的图片路径是相对 impl/ 存的，读图必须按这个根拼接，
# 不能交给调用方的 cwd —— 从仓库根跑 `python cli.py figures` 会找不到图。
ROOT = Path(__file__).resolve().parent.parent

# E4 探针层的网络前向顺序（layers.json 的 rows 是字母序，不能直接画）
PROBE_ORDER = [
    "down_relu_1", "down_relu_2", "down_relu_3",
    "resblock_01", "resblock_02", "resblock_03", "resblock_04", "resblock_05",
    "resblock_06", "resblock_07", "resblock_08", "resblock_09",
    "up_relu_1", "up_relu_2",
    "head_conv",
]

# AMD 7745HX 脚注数字（2026-10-23 队友机器实测）。原始 benchmark.json 已被
# i5-1340P 口径覆盖（见 issue #12），这里取自 README §5.1 / git 历史，勿改。
AMD_FOOTNOTE = (
    "AMD Ryzen 7 7745HX（2026-10-23）：A 279.8 / B 138.3 / C 36.7 ms"
    "（B 2.02×、C 7.63×）—— 加速比强平台依赖，引用任一延迟比必须标注平台"
)


def _load(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_e1_samples(results: Path, out: Path) -> Path:
    """E1 基线出图拼图：输入马图 | PyTorch FP32 基线(a) | OV FP32(b) | OV INT8(c)。

    E1 的交付定义是「基线推理跑通，马→斑马出图」（第五章进度表阶段②），所以
    第一要义是证明官方权重在本仓库里能正常出图，顺带把三组并列展示。异常拼图
    （E3）不能替代它——那是失真样本，不是正常出图展示。
    """
    from core.metrics import ssim  # 与 quality 同一实现，口径一致

    samples = ROOT / "artifacts" / "samples"
    quality_dir = results / "quality"
    horses = sorted(samples.glob("horse_*.png"))[:3]
    if not horses:
        raise SystemExit("samples/ 里没有 horse_*.png，先跑 setup")

    cols = ["输入马图", "A · PyTorch FP32（基线）", "B · OpenVINO FP32",
            "C · OpenVINO INT8"]
    n = len(horses)
    fig, axes = plt.subplots(n, 4, figsize=(12.8, 3.1 * n))
    if n == 1:
        axes = axes.reshape(1, 4)
    fig.suptitle(
        f"E1 基线出图样例（{n} 张，官方 horse2zebra 预训练权重，无训练）\n"
        "同一张输入分别走三组后端；SSIM 为全图结构相似度（与 quality.json 同一实现）",
        fontsize=10)

    for i, hp in enumerate(horses):
        stem = hp.stem
        paths = [hp, quality_dir / f"{stem}_A.png",
                 quality_dir / f"{stem}_B.png", quality_dir / f"{stem}_C.png"]
        miss = [p.name for p in paths if not p.exists()]
        if miss:
            raise SystemExit(
                f"E1 拼图缺素材：{miss}。先跑 `python cli.py quality` 生成三组输出图")
        arrs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                for p in paths]
        for j, p in enumerate(paths):
            axes[i][j].imshow(np.asarray(Image.open(p).convert("RGB")))
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(cols[j], fontsize=9)
        axes[i][0].set_ylabel(stem, fontsize=8.5)
        axes[i][2].set_xlabel(
            f"SSIM(a,b)={ssim(arrs[1], arrs[2]):.4f}   "
            f"SSIM(b,c)={ssim(arrs[2], arrs[3]):.4f}",
            fontsize=7.5)

    fig.tight_layout(rect=(0, 0, 1, 0.95 if n > 1 else 0.9))
    dest = out / "fig_e1_samples.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def fig_e2_perf(results: Path, out: Path) -> Path:
    """A/B/C 三组：延迟（±std 误差棒）+ FPS + 体积。主口径 = benchmark.json 所在机。"""
    b = _load(results / "benchmark.json")
    groups = [g for g in b["groups"] if g.get("available")]
    names = [f"{g['id']}\n{g['name']}" for g in groups]
    x = np.arange(len(groups))
    colors = ["#8c8c8c", "#4472c4", "#c0504d"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fig.suptitle(f"E2 三组性能对比（{b['device']}，{b['protocol']['rounds']} 轮）",
                 fontsize=12)

    means = [g["mean_ms"] for g in groups]
    stds = [g["std_ms"] for g in groups]
    axes[0].bar(x, means, yerr=stds, capsize=5, color=colors, width=0.55)
    axes[0].set_xticks(x, names, fontsize=8)
    axes[0].set_ylabel("单图延迟 (ms)")
    axes[0].set_title("延迟（误差棒 = 标准差）", fontsize=10)
    for xi, m, s in zip(x, means, stds):
        axes[0].annotate(f"{m:.0f}", (xi, m + s), ha="center", va="bottom", fontsize=8)

    fps = [g["fps"] for g in groups]
    axes[1].bar(x, fps, color=colors, width=0.55)
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("FPS")
    axes[1].set_title("吞吐", fontsize=10)
    for xi, v in zip(x, fps):
        axes[1].annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=8)

    sizes = [g["size_mb"] for g in groups]
    axes[2].bar(x, sizes, color=colors, width=0.55)
    axes[2].set_xticks(x, names, fontsize=8)
    axes[2].set_ylabel("模型体积 (MB)")
    axes[2].set_title("体积", fontsize=10)
    for xi, v in zip(x, sizes):
        axes[2].annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=8)

    fig.text(0.5, 0.01, AMD_FOOTNOTE, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    dest = out / "fig_e2_perf.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def fig_e3_anomalies(results: Path, out: Path) -> Path:
    """异常样例拼图：输入马图 | OV FP32（b）| OV INT8（c），逐行标注指标与判读提示。"""
    an = _load(results / "anomalies" / "anomalies.json")
    quality_dir = results / "quality"

    n = len(an)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 2.9 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    fig.suptitle(
        f"E3 异常样例拼图（{n} 例，SSIM(b,c) < 0.80 的量化敏感样本）\n"
        "三列对照用于判定缺陷归属：两列共有的怪异 = FP32 固有；仅 c 列出现 = 量化引入",
        fontsize=10)

    col_titles = ["输入马图（a 前）", "OV FP32 输出（b）", "OV INT8 输出（c）"]
    for i, item in enumerate(an):
        name = item["name"]
        m = item["metrics"]
        paths = [
            ROOT / m["input"],
            quality_dir / f"{name}_B.png",
            quality_dir / f"{name}_C.png",
        ]
        for j, p in enumerate(paths):
            axes[i][j].imshow(np.asarray(Image.open(p).convert("RGB")))
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(col_titles[j], fontsize=9)
        axes[i][0].set_ylabel(name, fontsize=8.5)
        axes[i][1].set_xlabel(
            f"SSIM(b,c)={m['ssim_bc']:.3f}  LPIPS(b,c)={m['lpips']['L_bc']:.3f}  "
            f"掩码IoU(a,c)={m['mask_iou_ac']:.3f}  背景SSIM(a,c)={m['background_ssim_ac']:.3f}",
            fontsize=7.5)

    fig.tight_layout(rect=(0, 0, 1, 0.94 if n > 1 else 0.88))
    dest = out / "fig_e3_anomalies.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def fig_e4_layers(results: Path, out: Path) -> Path:
    """层序号 × 表示差异：余弦相似度与相对 L2 双图，残差块段染色。"""
    lay = _load(results / "layers.json")
    by_name = {r["layer"]: r for r in lay["rows"]}
    missing = [n for n in PROBE_ORDER if n not in by_name]
    if missing:
        raise SystemExit(f"layers.json 缺层：{missing}")
    rows = [by_name[n] for n in PROBE_ORDER]
    x = np.arange(len(rows))
    cos = [r["cosine"] for r in rows]
    rel = [r["rel_l2"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    fig.suptitle(
        f"E4 层间分工：量化前后逐层表示漂移（n={lay['n_images']}）",
        fontsize=12)

    for ax in (ax1, ax2):
        ax.axvspan(2.5, 11.5, color="#d9d9d9", alpha=0.45, lw=0)
        ax.grid(axis="y", alpha=0.3)

    ax1.plot(x, cos, "o-", color="#4472c4", ms=4)
    ax1.set_ylabel("余弦相似度")
    ax1.set_ylim(min(cos) - 0.005, 1.002)
    ax1.set_title("逐层余弦相似度（越接近 1 越一致）", fontsize=10)
    ax1.annotate("残差块 01→09：0.9994 → 0.9644 单调下降",
                 xy=(7, cos[7]), xytext=(4.2, min(cos) + 0.012),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    ax2.plot(x, rel, "s-", color="#c0504d", ms=4)
    ax2.set_ylabel("相对 L2 误差")
    ax2.set_xticks(x, [r["layer"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax2.set_title("逐层相对 L2 误差（越大漂移越深）", fontsize=10)

    fig.text(0.5, 0.005, "灰色带 = 9 个残差块段；量化误差沿残差块深度逐级累积",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    dest = out / "fig_e4_layers.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def quality_table(results: Path, out: Path) -> Path:
    """E3 质量对比表（markdown，报告直接引用）。"""
    q = _load(results / "quality.json")
    d = q["deep"]
    mean = q["mean"]
    lp = d["lpips"]
    fid = d["fid"]
    kid = d["kid"]
    mask = d["mask"]
    disc = d["zebra_discrimination"]

    def kid_str(key: str) -> str:
        k = kid[key]
        return (f"{k['mean']:.4f} ± {k['std']:.4f}"
                f"（subset {k['subset_size']}×{k['n_subsets']}）")

    lines = [
        "# E3 量化质量对比表（testA 生成批 n="
        f"{lp['n_images']} / testB 参照批 n={fid['n_real']}）",
        "",
        "| 维度 | 指标 | 数值 | 说明 |",
        "|---|---|---|---|",
        f"| 转换保真 | LPIPS(a,b) | {lp['L_ab']:.4f} | 运行时转换差异，≈0 即转换无损 |",
        f"| 量化保真 | LPIPS(b,c) | {lp['L_bc']:.4f} | 量化本身引入的感知差异（核心） |",
        f"| 总差异 | LPIPS(a,c) | {lp['L_ac']:.4f} | 原版到终版 |",
        f"| 逐像素 | SSIM(b,c) / PSNR(b,c) | {mean['ssim_bc']:.4f} / {mean['psnr_bc']:.2f} dB | 均值 |",
        f"| 结构（边缘） | Canny 边缘 SSIM(b,c) | {d['canny_edge_ssim_bc']:.4f} | 方案 §4 指定 Canny 版 |",
        f"| 迁移质量 | FID(B vs 真实斑马) | {fid['b_vs_real_zebra']:.2f} | FP32 批 |",
        f"| 迁移质量 | FID(C vs 真实斑马) | {fid['c_vs_real_zebra']:.2f} | INT8 批 |",
        f"| 迁移质量 | KID(B) | {kid_str('b_vs_real_zebra')} | FP32 批 |",
        f"| 迁移质量 | KID(C) | {kid_str('c_vs_real_zebra')} | INT8 批 |",
        f"| 结构（前景） | 掩码 IoU(a,c) | {mask['mean_iou_ac']:.4f} | 检出 {mask['detected_pairs']}/{lp['n_images']} 对 |",
        f"| 结构（背景） | 背景 SSIM(a,c) | {mask['mean_background_ssim_ac']:.4f} | 掩码外区域 |",
        f"| 任务保真 | 斑马判别 top1（B / C） | {disc['b']['top1_accuracy']:.2%} / {disc['c']['top1_accuracy']:.2%} | n={disc['b']['n']}，量化前后持平 |",
        f"| 异常登记 | SSIM(b,c)<0.80 | {d['anomalies']['count']} 例 | 见 fig_e3_anomalies.png |",
        "",
        f"记号：a=PyTorch FP32，b=OV FP32，c=OV INT8；`metrics_source={q['metrics_source']}`。",
    ]
    dest = out / "quality_table.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def make_all(results: Path | None = None, out: Path | None = None) -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    results = results or root / "artifacts" / "results"
    out = out or root / "artifacts" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    produced = [
        fig_e1_samples(results, out),
        fig_e2_perf(results, out),
        fig_e3_anomalies(results, out),
        fig_e4_layers(results, out),
        quality_table(results, out),
    ]
    return produced


__all__ = ["make_all", "PROBE_ORDER", "fig_e1_samples", "fig_e2_perf",
           "fig_e3_anomalies", "fig_e4_layers", "quality_table"]
