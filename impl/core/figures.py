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
import math
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


def _pair(pair=None):
    """迁移对对象；支持传入 Pair、pair_id 字符串或不传（用默认 horse2zebra）。"""
    from core.pairs import DEFAULT_PAIR, load_pair
    if pair is None:
        return load_pair(DEFAULT_PAIR)
    if isinstance(pair, str):
        return load_pair(pair)
    return pair


def fig_e1_samples(results: Path, out: Path, pair=None) -> Path:
    """E1 基线出图拼图：输入图 | PyTorch FP32 基线(a) | OV FP32(b) | OV INT8(c)。

    E1 的交付定义是「基线推理跑通，输入域 → 目标域出图」（第五章进度表阶段②），
    所以第一要义是证明官方权重在本仓库里能正常出图，顺带把三组并列展示。异常拼图
    （E3）不能替代它——那是失真样本，不是正常出图展示。
    """
    from core.metrics import ssim  # 与 quality 同一实现，口径一致

    p = _pair(pair)
    samples = p.paths.samples
    eval_in = p.paths.eval_in
    quality_dir = results / "quality"

    # 素材来源：quality 产出的三组图决定"哪些图有输出"；
    # 输入原图优先取演示样例，样例没有就退回该 pair 评测集（输入域）。
    # 这样 E1 不要求 quality 一定跑在样例图上（正式口径跑的是评测集全量）。
    inputs = []
    for a in sorted(quality_dir.glob("*_A.png")):
        stem = a.name[: -len("_A.png")]
        src = None
        for d in (samples, eval_in):
            for ext in (".png", ".jpg", ".jpeg"):
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    src = cand
                    break
            if src:
                break
        if src is None:
            continue
        if all((quality_dir / f"{stem}_{g}.png").exists() for g in "ABC"):
            inputs.append(src)
        if len(inputs) >= 3:
            break
    if not inputs:
        raise SystemExit(
            f"E1 拼图缺素材：{quality_dir} 下没有同时具备 A/B/C 三组输出的图。"
            f"先跑 `python cli.py quality --pair {p.id}` 生成三组输出图")

    cols = [f"输入图（{p.input_prefix}）", "A · PyTorch FP32（基线）",
            "B · OpenVINO FP32", "C · OpenVINO INT8"]
    n = len(inputs)
    fig, axes = plt.subplots(n, 4, figsize=(12.8, 3.1 * n))
    if n == 1:
        axes = axes.reshape(1, 4)
    fig.suptitle(
        f"E1 基线出图样例（{n} 张，官方 {p.id} 预训练权重，无训练）\n"
        "同一张输入分别走三组后端；SSIM 为全图结构相似度（与 quality.json 同一实现）",
        fontsize=10)

    for i, hp in enumerate(inputs):
        stem = hp.stem
        paths = [hp, quality_dir / f"{stem}_A.png",
                 quality_dir / f"{stem}_B.png", quality_dir / f"{stem}_C.png"]
        miss = [p.name for p in paths if not p.exists()]
        if miss:
            from core.backends import Runtime
            from core.utils import load_image, to_tensor, to_image
            models = ROOT / 'artifacts/models'
            rt = Runtime(models / 'horse2zebra.pth', models / 'horse2zebra.xml',
                         models / 'horse2zebra_int8.xml')
            if not all(rt.availability().values()):
                raise SystemExit(f'E1 缺素材 {miss}，且 A/B/C 模型不完整；先运行 setup')
            quality_dir.mkdir(parents=True, exist_ok=True)
            tensor = to_tensor(load_image(hp))
            for group, dest in zip(('A', 'B', 'C'), paths[1:]):
                if not dest.exists():
                    to_image(rt.infer(tensor, group)).save(dest)
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
    """异常样例拼图：输入图 | OV FP32（b）| OV INT8（c），逐行标注指标与判读提示。"""
    idx = results / "anomalies" / "anomalies.json"
    if not idx.exists():
        raise SystemExit(
            f"没有异常登记（{idx}）：异常表由 `quality --deep` 生成，先跑一次深度评测")
    an = _load(idx)
    quality_dir = results / "quality"
    if not an:
        raise SystemExit("异常登记为 0 例，无需出图（这本身是结论，如实记录即可）")

    n = len(an)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        ax.text(.5, .5, '本次无 SSIM 阈值异常；不代表全部样本已人工复核', ha='center')
        dest = out / 'fig_e3_anomalies.png'
        fig.savefig(dest, dpi=200)
        plt.close(fig)
        return dest
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
            f"输出IoU(a,c)={_metric_text(m.get('mask_iou_ac'))}  旧背景SSIM(a,c)={_metric_text(m.get('background_ssim_ac'))}",
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
    ax1.annotate(f"残差块 01→09：{by_name['resblock_01']['cosine']:.4f} → {by_name['resblock_09']['cosine']:.4f}",
                 xy=(7, cos[7]), xytext=(4.2, min(cos) + 0.012),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    ax2.plot(x, rel, "s-", color="#c0504d", ms=4)
    ax2.set_ylabel("相对 L2 误差")
    ax2.set_xticks(x, [r["layer"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax2.set_title("逐层相对 L2 误差（越大漂移越深）", fontsize=10)

    fig.text(0.5, 0.005, "灰色带 = 9 个残差块段；表示漂移不等同于语义因果归因",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    dest = out / "fig_e4_layers.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def _is_num(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _is_pos_int(value) -> bool:
    return type(value) is int and value > 0


def _is_nonneg_int(value) -> bool:
    return type(value) is int and value >= 0


def _metric_text(value, digits=3):
    return '缺失' if not _is_num(value) else f'{value:.{digits}f}'


def quality_table(results: Path, out: Path, pair=None) -> Path:
    """E3 质量对比表（markdown，报告直接引用）。"""
    from core.quality_state import format_accuracy_pct, resolve_classification

    p = _pair(pair)
    ref = p.ref_prefix           # 参照域（zebra / horse / monet / winter）
    q = _load(results / "quality.json")
    if (not isinstance(q, dict)
            or not isinstance(q.get("deep"), dict)
            or q.get("metrics_source") not in ("real", "partial")):
        raise SystemExit(
            f"{p.id} 的 quality.json 没有 deep 字段（metrics_source="
            f"{q.get('metrics_source') if isinstance(q, dict) else None}）："
            f"本表需要真实感知指标，先跑 `python cli.py quality --pair {p.id} --deep`")
    d = q["deep"]
    mean = q.get("mean") if isinstance(q.get("mean"), dict) else {}
    lp = d.get("lpips") if isinstance(d.get("lpips"), dict) else {}
    fid = d.get("fid") if isinstance(d.get("fid"), dict) else {}
    kid = d.get("kid") if isinstance(d.get("kid"), dict) else {}

    lp_n = lp.get("n_images") if _is_pos_int(lp.get("n_images")) else None
    q_n = q.get("n_samples") if _is_pos_int(q.get("n_samples")) else None
    if lp_n is not None:
        gen_head = f"生成批 n={lp_n}"
    elif q_n is not None:
        gen_head = f"输入总数 n={q_n}"
    else:
        gen_head = "样本量未记录"
    mask_total = lp_n if lp_n is not None else (q_n if q_n is not None else "未记录")

    n_real = fid.get("n_real") if _is_pos_int(fid.get("n_real")) else None
    ref_head = f"{ref} 参照批 n={n_real}" if n_real is not None else f"{ref} 参照数未记录"
    k_b, k_c = f"b_vs_real_{ref}", f"c_vs_real_{ref}"

    def kid_str(key: str) -> str:
        k = kid.get(key) if isinstance(kid, dict) else None
        if not isinstance(k, dict) or not _is_num(k.get("mean")):
            return "缺失"
        std_part = f" ± {k['std']:.4f}" if _is_num(k.get("std")) else "（标准差未记录）"
        sub_sz = k.get("subset_size") if _is_pos_int(k.get("subset_size")) else None
        n_sub = k.get("n_subsets") if _is_pos_int(k.get("n_subsets")) else None
        sub_part = f"（subset {sub_sz}×{n_sub}）" if (sub_sz is not None and n_sub is not None) else "（子集未记录）"
        return f"{k['mean']:.4f}{std_part}{sub_part}"

    ssim_txt = _metric_text(mean.get("ssim_bc"), 4)
    psnr_txt = f"{mean['psnr_bc']:.2f} dB" if _is_num(mean.get("psnr_bc")) else "缺失"
    pixel_txt = "缺失" if (ssim_txt == "缺失" and psnr_txt == "缺失") else f"{ssim_txt} / {psnr_txt}"

    lines = [
        "# E3 量化质量对比表（"
        f"{p.id}：{gen_head} / {ref_head}）",
        "",
        "| 维度 | 指标 | 数值 | 说明 |",
        "|---|---|---|---|",
        f"| 转换保真 | LPIPS(a,b) | {_metric_text(lp.get('L_ab'), 4)} | 运行时转换差异，≈0 即转换无损 |",
        f"| 量化保真 | LPIPS(b,c) | {_metric_text(lp.get('L_bc'), 4)} | 量化本身引入的感知差异（核心） |",
        f"| 总差异 | LPIPS(a,c) | {_metric_text(lp.get('L_ac'), 4)} | 原版到终版 |",
        f"| 逐像素 | SSIM(b,c) / PSNR(b,c) | {pixel_txt} | 均值 |",
        f"| 结构（边缘） | Canny 边缘 SSIM(b,c) | {_metric_text(d.get('canny_edge_ssim_bc'), 4)} | 方案 §4 指定 Canny 版 |",
        f"| 迁移质量 | FID(B vs 真实{ref}) | {_metric_text(fid.get(k_b), 2)} | FP32 批 |",
        f"| 迁移质量 | FID(C vs 真实{ref}) | {_metric_text(fid.get(k_c), 2)} | INT8 批 |",
        f"| 迁移质量 | KID(B) | {kid_str(k_b)} | FP32 批 |",
        f"| 迁移质量 | KID(C) | {kid_str(k_c)} | INT8 批 |",
    ]
    # 结构 / 任务保真两维按 pair manifest 开关条件渲染：关闭的 pair 显示「不适用」
    # 而启用但缺失结果的 pair 显示「缺失」，不混淆两者。
    if not p.metric_enabled("mask_iou"):
        mask_reason = p.metric_reason("mask_iou") or f"{p.id} 未启用 mask_iou（见 pairs/*.json）"
        lines.append(
            f"| 结构（前景/背景） | 掩码 IoU / Dice / 背景 SSIM | 不适用 | {mask_reason} |")
    else:
        mask = d.get("mask") if isinstance(d.get("mask"), dict) else {}
        if mask:
            det_pairs = mask["detected_pairs"] if _is_nonneg_int(mask.get("detected_pairs")) else "未记录"
            dice_n = det_pairs if "mean_dice_ac" in mask else "未记录"
            denom_iou = f"检出 {det_pairs}/{mask_total} 对" if mask_total != "未记录" else f"检出 {det_pairs}（总输入数未记录）"
            denom_dice = f"有效 {dice_n}/{mask_total} 对" if mask_total != "未记录" else f"有效 {dice_n}（总输入数未记录）"
            lines += [
                f"| 结构（前景） | 掩码 IoU(a,c) | {_metric_text(mask.get('mean_iou_ac'), 4)} | {denom_iou} |",
                f"| 结构（前景） | 掩码 Dice(a,c) | {_metric_text(mask.get('mean_dice_ac'), 4)} | {denom_dice}；旧 JSON 未计算时显示缺失 |",
                f"| 结构（背景） | 背景 SSIM(a,c) | {_metric_text(mask.get('mean_background_ssim_ac'), 4)} | 掩码外区域 |",
            ]
        else:
            lines.append(
                "| 结构（前景/背景） | 掩码 IoU / Dice / 背景 SSIM | 缺失 | 缺失（未取得有效掩码结果） |")

    cls = resolve_classification(p, q)
    cls_status = cls["status"]
    cls_target = cls["target"]
    b_batch = cls["batches"]["b"]
    c_batch = cls["batches"]["c"]

    def _batch_n_str(label: str, item: dict | None) -> str:
        if not item:
            return f"缺少 {label} 批结果"
        return f"{label} 批 n={item['n']}" if item.get("n") is not None else f"{label} 批样本量未记录"

    if cls_status == "disabled":
        lines.append(
            f"| 任务保真 | 类别判别 | 不适用 | 类别判别不适用：{cls['reason']} |")
    elif cls_status == "complete" and b_batch is not None and c_batch is not None:
        b_pct = format_accuracy_pct(b_batch["top1_accuracy"])
        c_pct = format_accuracy_pct(c_batch["top1_accuracy"])
        lines.append(
            f"| 任务保真 | {cls_target} 判别 top1（B / C） | {b_pct} / {c_pct} | "
            f"{_batch_n_str('B', b_batch)}，{_batch_n_str('C', c_batch)}；差异应据实解释 |")
    elif cls_status == "partial":
        b_cell = format_accuracy_pct(b_batch["top1_accuracy"]) if b_batch is not None else "缺失"
        c_cell = format_accuracy_pct(c_batch["top1_accuracy"]) if c_batch is not None else "缺失"
        lines.append(
            f"| 任务保真 | {cls_target} 判别 top1（B / C） | {b_cell} / {c_cell} | "
            f"{_batch_n_str('B', b_batch)}；{_batch_n_str('C', c_batch)} |")
    else:
        has_err = isinstance(d.get("errors"), list) and bool(d["errors"])
        err_note = "（请查 deep.errors）" if has_err else ""
        lines.append(
            f"| 任务保真 | {cls_target} 判别 top1（B / C） | 缺失 | 此次未取得有效类别判别结果{err_note} |")

    anom = d.get("anomalies") if isinstance(d.get("anomalies"), dict) else {}
    anom_count = anom.get("count")
    anom_txt = f"{anom_count} 例" if _is_nonneg_int(anom_count) else "未记录"
    lines += [
        f"| 异常登记 | SSIM(b,c)<0.80 | {anom_txt} | 见 fig_e3_anomalies.png |",
        "",
        f"记号：a=PyTorch FP32，b=OV FP32，c=OV INT8；`metrics_source={q['metrics_source']}`。",
    ]
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "quality_table.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if isinstance(d.get("structure"), dict) and isinstance(d["structure"].get("pairs"), dict):
        structure_table(q, out)
    return dest


def structure_table(q: dict, out: Path) -> Path:
    """v2 口径独立成表，明确分母与人工复核状态。"""
    report = q['deep']['structure']
    lines = ['# 结构评测 v2', '',
             'x=输入马图；a/b/c=三组生成图。掩码为检测器估计，不是人工真值。',
             'IoU/Dice 采用各侧最高置信度目标；多实例对应关系须人工复核。',
             '共同背景仅平均两侧前景并集外的完整 11×11 SSIM 窗口，至少 64 窗口。', '',
             '| 比较 | 指标 | 均值 | 有效/总样本 |', '|---|---|---:|---:|']
    for pair, values in report['pairs'].items():
        label = ('输入保真 ' if pair.startswith('x') else '输出一致性 ') + pair
        for metric, item in values.items():
            value = '缺失' if item['mean'] is None else f"{item['mean']:.6f}"
            lines.append(f"| {label} | {metric} | {value} | {item['n_valid']}/{item['n_total']} |")
    lines += ['', f"检测失败样本：{report['n_failure_samples']}；待复核样本：{len(report['review_queue'])}。",
              '人工复核尚未完成。缺失值不填零；异常样本不从全量指标中剔除。',
              '分类器使用 ResNet50 IMAGENET1K_V1 配套预处理；不可直接混用旧判别率。',
              'FID 沿用历史 torchvision 特征实现，不与外部标准 FID 数值直接比较。']
    dest = out / 'structure_table_v2.md'
    dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dest


def make_all(results: Path | None = None, out: Path | None = None,
             pair=None) -> list[Path]:
    """生成全部图表。单张图缺素材时只跳过该图，不拖垮其余图。"""
    p = _pair(pair)
    results = results or p.paths.results
    out = out or results.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    steps = [
        ("E1 基线出图", lambda: fig_e1_samples(results, out, p)),
        ("E2 三组性能", lambda: fig_e2_perf(results, out)),
        ("E3 异常拼图", lambda: fig_e3_anomalies(results, out)),
        ("E4 层间曲线", lambda: fig_e4_layers(results, out)),
        ("E3 质量对比表", lambda: quality_table(results, out, p)),
    ]
    produced: list[Path] = []
    for name, fn in steps:
        try:
            produced.append(fn())
        except SystemExit as e:
            print(f"  [跳过] {name}：{e}")
        except Exception as e:
            print(f"  [跳过] {name}：{type(e).__name__}: {e}")
    return produced


__all__ = ["make_all", "PROBE_ORDER", "fig_e1_samples", "fig_e2_perf",
           "fig_e3_anomalies", "fig_e4_layers", "quality_table"]
