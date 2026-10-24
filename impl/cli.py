"""ShiftLite 命令行入口。

    python cli.py setup      # 下载模型 → 转 ONNX → 转 IR → INT8 量化 → 拉样例图
    python cli.py bench      # A/B/C 三组性能对比（延迟 / FPS / 体积）
    python cli.py quality    # 三组输出的质量与结构保真度量
    python cli.py serve      # 启动本地可视化界面
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
MODELS = ART / "models"
SAMPLES = ART / "samples"
RESULTS = ART / "results"

# 权重的下载源（按优先级逐个尝试）。
# PLAN §3.3 写的是"双源"，但实际只 anchor 得住一个：伯克利直链 200 OK、
# content-length 45575747 与 MODEL_BYTES 完全对得上，是验过的；
# HuggingFace 上没找到能确认可用的同名权重仓库（HF API 在本环境不可达，
# 编一个 URL 进去只会让人白等），所以暂不列第二源。要用备源就用
# 环境变量 SHIFTLITE_WEIGHTS_URL 指过去，它会被排在最前面。
MODEL_SOURCES = [
    ("伯克利官方", "http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/"
                   "horse2zebra.pth"),
]
MODEL_BYTES = 45575747


def model_sources() -> list[tuple[str, str]]:
    override = os.environ.get("SHIFTLITE_WEIGHTS_URL", "").strip()
    if override:
        return [("环境变量 SHIFTLITE_WEIGHTS_URL", override)] + MODEL_SOURCES
    return MODEL_SOURCES

# 样例图与校准集来源（按优先级尝试；mini 只有 1.8MB，几秒下完）
DATASET_SOURCES = [
    ("horse2zebra.zip", "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/"
                        "horse2zebra.zip", 111),
    ("mini.zip", "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/"
                 "mini.zip", 1.8),
]

CALIB = ART / "calib"

# 方案 §3.3 口径：校准集取 trainA 前 300 张马图，与评测所用 test 集隔离；
# 演示/评测样例取 testA、testB。三者来源不同，不可互相替代。
CALIB_SUBSET = 300
N_SAMPLE_HORSE = 6
N_SAMPLE_ZEBRA = 3


def _p(msg: str) -> None:
    print(msg, flush=True)


# ---------------- setup ----------------

def _resume_download(url: str, dst: Path, validator=None, attempts: int = 40) -> Path:
    """断点续传；validator 返回 True 表示文件已完整。"""
    have = dst.stat().st_size if dst.exists() else 0
    for attempt in range(1, attempts + 1):
        if validator and validator(dst):
            return dst
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                mode = "ab" if have else "wb"
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
        except Exception as e:
            _p(f"  第 {attempt} 次中断（{type(e).__name__}），已下 "
               f"{have/1024/1024:.1f} MB，续传…")
            time.sleep(1)
            have = dst.stat().st_size if dst.exists() else 0
            continue
        have = dst.stat().st_size
    if not (validator and validator(dst)):
        raise SystemExit(f"下载未完成：{dst}（{have} 字节）")
    return dst


def download_model(force: bool = False) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    dst = MODELS / "horse2zebra.pth"
    ok = lambda p: p.exists() and p.stat().st_size >= MODEL_BYTES
    if ok(dst) and not force:
        _p(f"  [跳过] 权重已存在 {dst.stat().st_size/1024/1024:.1f} MB")
        return dst
    _p("  下载预训练权重（支持断点续传）…")
    for label, url in model_sources():
        try:
            _p(f"  尝试 {label}…")
            _resume_download(url, dst, ok)
            if ok(dst):
                _p(f"  [完成] {dst} ({dst.stat().st_size/1024/1024:.1f} MB)"
                   f"，来源：{label}")
                return dst
            _p(f"  [放弃] {label} 下载不完整，换下一个源")
        except SystemExit:
            _p(f"  [放弃] {label} 不可用，换下一个源")
            continue
    raise SystemExit(
        "权重下载失败：所有镜像都不可用。手动下载 horse2zebra.pth 放到 "
        f"{dst} 后重跑 setup 即可（完整大小 {MODEL_BYTES/1024/1024:.1f} MB）。")


def _zip_ok(path: Path) -> bool:
    """只判断能不能用：能打开、条目够多、A/B 两个域都在。

    不做全量 CRC 校验——断点续传偶尔会让个别不用的条目损坏，
    那种情况下仍应继续（读取失败的条目在解压时跳过）。
    """
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) < 50:
                return False
            return any("/testA/" in n for n in names) and any(
                "/testA/" in n or "/trainA/" in n for n in names)
    except Exception:
        return False


def dir_prefix(names) -> str:
    """一批样本在 zip 里的目录前缀（如 horse2zebra/trainA），用于记录数据来源。"""
    for n in names:
        return "/".join(n.split("/")[:-1]) or "."
    return "-"


def download_dataset() -> Path:
    """取数据集：先复用已下过的包，没有则按 DATASET_SOURCES 顺序下载。

    顺序以「能否支撑方案 §3.3 的口径」为准，不看下载快慢：校准要 trainA
    前 300 张、评测要 testA 120 / testB 140 张，只有 111 MB 的完整包满足；
    mini 包各域仅 10 张（实测 45 个条目），只能用来跑通流程，不能出结论数据。
    """
    # 1) 已有且完整的包直接复用，完整版优先
    for fname, _url, _size in DATASET_SOURCES:
        dst = ART / fname
        if _zip_ok(dst):
            _p(f"  [复用] {fname}（{dst.stat().st_size/1024/1024:.1f} MB）")
            return dst

    # 2) 都没下过：从完整包开始试
    for fname, url, size_mb in DATASET_SOURCES:
        dst = ART / fname
        try:
            _p(f"  下载 {fname}（约 {size_mb} MB，支持断点续传）…")
            _resume_download(url, dst, _zip_ok,
                             attempts=20 if size_mb < 10 else 4)
            _p(f"  [完成] {fname}（{dst.stat().st_size/1024/1024:.1f} MB）")
            return dst
        except SystemExit:
            _p(f"  [放弃] {fname} 未下完，换下一个源")
            continue
    raise SystemExit("样例数据全部下载失败，请检查网络后重试")


def prepare_samples(n_horse: int = N_SAMPLE_HORSE, n_zebra: int = N_SAMPLE_ZEBRA,
                    n_calib: int = CALIB_SUBSET) -> int:
    """从数据集解压演示样例与校准集。

    三批数据的来源互不重叠，对应方案 §3.3 与 §4 的口径：
        samples/ 演示与评测用：testA（马）前 n_horse 张 + testB（斑马）前 n_zebra 张
        calib/   量化校准用：trainA 前 n_calib 张
    校准集必须取自 trainA；若与评测集同源（例如都取 testA），INT8 的量化参数
    会在待评测的那几张图上统计得到，SSIM/LPIPS 的 b-c 差值会被系统性地低估。
    """
    import zipfile
    import json

    SAMPLES.mkdir(parents=True, exist_ok=True)
    CALIB.mkdir(parents=True, exist_ok=True)
    zip_path = download_dataset()

    skipped: list[str] = []

    def save(z, name, dst):
        """解一张图。失败时不计入 used，并把文件名记下来——zip 条目损坏
        在断点续传后时有发生，静默跳过会让"准备了 300 张"变成假话。"""
        try:
            with z.open(name) as f:
                Image.open(f).convert("RGB").save(dst)
            return True
        except Exception as e:
            skipped.append(f"{name}（{type(e).__name__}）")
            return False

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # 样例优先取官方测试集；测试集缺失时才退回训练集
        horse_test = sorted(n for n in names
                            if "/testA/" in n and n.lower().endswith(".jpg"))
        horse_train = sorted(n for n in names
                             if "/trainA/" in n and n.lower().endswith(".jpg"))
        zebra_test = sorted(n for n in names
                            if "/testB/" in n and n.lower().endswith(".jpg"))
        zebra_train = sorted(n for n in names
                             if "/trainB/" in n and n.lower().endswith(".jpg"))
        horses = horse_test or horse_train
        zebras = zebra_test or zebra_train
        if not horses:
            raise SystemExit(f"{zip_path.name} 里找不到马（A 域）图像")

        used = 0
        for i, name in enumerate(horses[:n_horse], 1):
            if save(z, name, SAMPLES / f"horse_{i:02d}.png"):
                used += 1
        for i, name in enumerate(zebras[:n_zebra], 1):
            if save(z, name, SAMPLES / f"zebra_{i:02d}.png"):
                used += 1

        # 校准集：trainA 前 n_calib 张（方案 §3.3），与 testA/testB 评测集隔离。
        if not horse_train:
            raise SystemExit(
                f"{zip_path.name} 里没有 trainA（训练集马图），无法准备校准集。\n"
                f"        请删除 {zip_path} 后重跑 setup 以下载完整版 horse2zebra.zip。")
        calib_src = horse_train
        calib_n = min(len(calib_src), n_calib)
        written = []
        for i, name in enumerate(calib_src[:calib_n], 1):
            if save(z, name, CALIB / f"calib_{i:03d}.png"):
                written.append(name)

    # 留一份来源清单，报告里可以据此证明校准集与评测集不重叠
    (CALIB / "manifest.json").write_text(
        json.dumps({
            "source_zip": zip_path.name,
            "calib_prefix": dir_prefix(calib_src),
            "requested": n_calib,
            "written": len(written),
            "eval_samples_prefix": dir_prefix(horses[:n_horse]),
            "files": written[:10],
            "skipped_count": len(skipped),
            "skipped": skipped[:20],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    _p(f"  来源 {zip_path.name}：样例 {used} 张，校准集 {len(written)} 张")
    if skipped:
        _p(f"  ⚠️ {len(skipped)} 个 zip 条目解压失败已跳过（见 calib/manifest.json）")
    if len(written) < n_calib:
        _p(f"  ⚠️ 校准集只有 {len(written)} 张，方案 §3.3 要求 trainA 前 {n_calib} 张；"
           f"请在报告中如实写明实际张数，或重跑 setup 取完整数据集。")
    return used


def cmd_setup(args):
    _p("== 1/4 预训练权重 ==")
    weights = download_model(force=args.force)

    _p("== 2/4 样例图与校准集 ==")
    n = prepare_samples(n_calib=args.subset)
    _p(f"  样例 {n} 张")

    from core.convert import build_ir, verify_against_torch

    _p("== 3/4 PyTorch 直转 OpenVINO IR（不经 ONNX）==")
    xml = MODELS / "horse2zebra.xml"
    if xml.exists() and not args.force:
        _p(f"  [跳过] {xml.name}")
    else:
        build_ir(weights, xml)
        _p(f"  [完成] {xml.name}")

    # 判据要用真实图像：造出来的随机输入值域约 ±3，不是真实输入分布
    ref = sorted(SAMPLES.glob("horse_*.png"))
    report = verify_against_torch(weights, xml, image=ref[0] if ref else None)
    _p(f"  验证输入：{report.get('input_image') or '固定随机输入（样例图缺失）'}")
    _p(f"  转换正确性：max|Δ|={report['max_gray']} 灰度级，"
       f"mean|Δ|={report['mean_gray']} 灰度级"
       f"（判据 {'通过' if report['pass'] else '未通过'}：均值≤0.5、最大≤5）")
    _p(f"  IR 规模：{report['total_ops']} 个节点 / {report['op_kinds']} 类算子")
    (MODELS / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _p("== 4/4 NNCF INT8 量化 ==")
    int8 = MODELS / "horse2zebra_int8.xml"
    if int8.exists() and not args.force:
        _p(f"  [跳过] {int8.name}")
    else:
        from core.quantize import collect_calibration, quantize
        calib = collect_calibration(CALIB, args.subset)
        # 必须关键字传参：quantize 的第 4 个位置参数是 skip_instance_norm，
        # 不是 accuracy_control（实参错位会让默认 setup 产出崩溃的 INT8）。
        _, info = quantize(xml, calib, int8,
                           skip_instance_norm=not args.keep_instance_norm,
                           accuracy_control=args.accuracy_control)
        (MODELS / "quantize_report.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        _p(f"  [完成] {int8.name}（校准 {len(calib)} 张，mode={info['mode']}）")

    _p("\n全部就绪。下一步：python cli.py bench")


# ---------------- bench ----------------

def _cpu_name() -> str:
    if platform.system() == "Windows":
        import winreg
        try:
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def cmd_bench(args):
    from core.backends import Runtime, GROUP_LABEL
    from core.utils import SIZE

    rt = Runtime(MODELS / "horse2zebra.pth",
                 MODELS / "horse2zebra.xml",
                 MODELS / "horse2zebra_int8.xml")
    avail = rt.availability()
    if not any(avail.values()):
        raise SystemExit("没有任何可用模型，先跑 python cli.py setup")

    # 测速输入也换成真实图像：与时延相关的是计算量，用真实分布能顺带记录
    # 一次有代表性的输入；取不到样例图时退回固定随机输入。
    ref = sorted(SAMPLES.glob("horse_*.png"))
    if ref:
        from core.utils import load_image, to_tensor
        x = to_tensor(load_image(ref[0])).astype(np.float32)
        input_desc = ref[0].name
    else:
        rng = np.random.default_rng(20261014)
        x = rng.standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)
        input_desc = "固定随机输入"

    groups = []
    for gid in ["A", "B", "C"]:
        if not avail[gid]:
            groups.append({"id": gid, "name": GROUP_LABEL[gid],
                           "available": False, "reason": "模型未生成"})
            continue
        _p(f"  {gid} 组 {GROUP_LABEL[gid]} 预热…")
        rt.warmup(gid, args.warmup)

        times = []
        for i in range(args.rounds):
            t0 = time.perf_counter()
            rt.infer(x, gid)
            times.append((time.perf_counter() - t0) * 1000)
            _p(f"    第 {i+1}/{args.rounds} 轮 {times[-1]:.1f} ms")

        arr = np.array(times[args.warmup_drop:]) if len(times) > args.warmup_drop else np.array(times)
        mean = float(arr.mean())
        groups.append({
            "id": gid,
            "name": GROUP_LABEL[gid],
            "available": True,
            "mean_ms": round(mean, 2),
            "std_ms": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 2),
            "min_ms": round(float(arr.min()), 2),
            "max_ms": round(float(arr.max()), 2),
            "fps": round(1000 / mean, 2) if mean else 0,
            # rounds_total=实际计时轮次，rounds_used=丢弃 args.warmup_drop 轮
            # 之后参与统计的轮次；两个数分开记，避免读 JSON 的人猜。
            "rounds_total": int(len(times)),
            "rounds_used": int(arr.size),
            "size_mb": rt.model_size_mb(gid),
            # rt.precision() 返回的是运行时精度提示（三组都传了 f32 hint），
            # 直接落盘会让 C 组记成 float32 —— 界面是对的、产物文件是错的。
            # 这里按组别写真实数值格式，另用字段记录当时的提示值。
            "precision": "int8" if gid == "C" else "fp32",
            "inference_precision_hint": rt.precision(gid),
            "samples": [round(t, 2) for t in times],
        })

    base = next((g for g in groups if g["id"] == "A" and g.get("available")), None)
    for g in groups:
        if g.get("available") and base:
            g["speedup_vs_A"] = round(base["mean_ms"] / g["mean_ms"], 2)
            if g["size_mb"] and base["size_mb"]:
                g["size_ratio_vs_A"] = round(g["size_mb"] / base["size_mb"], 3)

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": _cpu_name(),
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "protocol": {
            "input": f"1×3×{SIZE}×{SIZE} float32（{input_desc}，[-1,1]）",
            "batch_size": 1,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "dropped_rounds": args.warmup_drop,
        },
        "groups": groups,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    _p("\n组别  均值(ms)  标准差  FPS    体积(MB)  相对A加速")
    for g in groups:
        if g.get("available"):
            _p(f"{g['id']}     {g['mean_ms']:>7.1f}  {g['std_ms']:>6.1f}  "
               f"{g['fps']:>6.1f}  {g['size_mb']:>7.2f}   {g.get('speedup_vs_A','-')}x")


# ---------------- quality ----------------

def cmd_quality(args):
    from core.backends import Runtime
    from core.metrics import (ssim, edge_ssim, mae, psnr, lpips_three_way)
    from core.utils import load_image, to_tensor, to_image

    rt = Runtime(MODELS / "horse2zebra.pth",
                 MODELS / "horse2zebra.xml",
                 MODELS / "horse2zebra_int8.xml")
    avail = rt.availability()

    out_dir = RESULTS / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    horses = sorted(p for p in SAMPLES.iterdir()
                    if p.suffix.lower() in exts and p.name.startswith("horse"))
    if not horses:
        raise SystemExit("样例目录里没有 horse_*.png，先跑 setup")

    rows = []
    YA, YB, YC = [], [], []      # deep 模式用的批量图像
    for p in horses[:args.limit]:
        img = load_image(p)
        x = to_tensor(img)
        y = {}
        for g in ["A", "B", "C"]:
            if not avail[g]:
                continue
            arr = rt.infer(x, g)
            out = to_image(arr)
            out.save(out_dir / f"{p.stem}_{g}.png")
            y[g] = np.asarray(out, dtype=np.float32)
        for g in ("A", "B", "C"):
            if g in y:
                {"A": YA, "B": YB, "C": YC}[g].append(y[g].astype(np.uint8))

        row = {"sample": p.stem, "input": str(p.relative_to(ROOT))}
        if "A" in y and "B" in y:
            row["ssim_ab"] = ssim(y["A"], y["B"])
            row["edge_ssim_ab"] = edge_ssim(y["A"], y["B"])
            row["mae_ab"] = mae(y["A"], y["B"])
            row["psnr_ab"] = psnr(y["A"], y["B"])
        if "B" in y and "C" in y:
            row["ssim_bc"] = ssim(y["B"], y["C"])
            row["edge_ssim_bc"] = edge_ssim(y["B"], y["C"])
            row["mae_bc"] = mae(y["B"], y["C"])
            row["psnr_bc"] = psnr(y["B"], y["C"])
        if "A" in y and "C" in y:
            row["ssim_ac"] = ssim(y["A"], y["C"])
            row["edge_ssim_ac"] = edge_ssim(y["A"], y["C"])

        if {"A", "B", "C"} <= y.keys():
            row["lpips"] = lpips_three_way(y["A"], y["B"], y["C"])

        rows.append(row)
        _p(f"  {p.stem}: " + "  ".join(
            f"{k}={v}" for k, v in row.items()
            if k.endswith(("_ab", "_bc", "_ac")) and isinstance(v, float)))

    def avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(float(np.mean(vals)), 6) if vals else None

    deep_block = {}
    if args.deep:
        deep_block = _deep_metrics(YA, YB, YC, rows, out_dir, args)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(rows),
        "metrics_source": "real" if deep_block else "proxy",
        "mean": {
            "ssim_ab": avg("ssim_ab"),
            "edge_ssim_ab": avg("edge_ssim_ab"),
            "psnr_ab": avg("psnr_ab"),
            "ssim_bc": avg("ssim_bc"),
            "edge_ssim_bc": avg("edge_ssim_bc"),
            "psnr_bc": avg("psnr_bc"),
            "ssim_ac": avg("ssim_ac"),
            "edge_ssim_ac": avg("edge_ssim_ac"),
        },
        "rows": rows,
    }
    if deep_block:
        summary["deep"] = deep_block
    (RESULTS / "quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"\n已写入 {RESULTS / 'quality.json'}"
       f"（{'真实感知指标' if deep_block else '代理值，仅供联调'}）")


def _deep_metrics(YA, YB, YC, rows, out_dir, args) -> dict:
    """接真实感知网络的一组指标。依赖缺失时整段跳过，不留半成品。"""
    try:
        from core.metrics_deep import (canny_edge_ssim, fid, inception_features,
                                       instance_masks, kid, lpips_three_way_real,
                                       register_anomalies, target_probabilities)
    except ImportError as e:
        _p(f"  [跳过] 深度指标依赖缺失：{e}")
        return {}

    import numpy as np

    from core.metrics import background_ssim, mask_iou
    from core.utils import load_image

    have = dict(A=bool(YA), B=bool(YB), C=bool(YC))
    deep: dict = {"requested": True, "groups": have}

    # 1) 真实 LPIPS 三对归因
    if all(have.values()):
        lp = lpips_three_way_real(YA, YB, YC)
        deep["lpips"] = {k: v for k, v in lp.items()
                         if not k.endswith("_per_image")}
        # 真实值覆盖 rows 里的代理 LPIPS，避免同一份 JSON 里两套口径
        for i, r in enumerate(rows):
            if i < len(YA):
                r["lpips"] = {
                    "L_ab": round(float(lp["L_ab_per_image"][i]), 6),
                    "L_bc": round(float(lp["L_bc_per_image"][i]), 6),
                    "L_ac": round(float(lp["L_ac_per_image"][i]), 6),
                    "note": "real",
                }
        _p(f"  LPIPS(真实) L_ab={lp['L_ab']}  L_bc={lp['L_bc']}  "
           f"L_ac={lp['L_ac']}")

    # 2) Canny 边缘 SSIM（方案指定，替代 metrics.py 的梯度幅值版本）
    for key, (x, y) in {"ab": (YA, YB), "bc": (YB, YC), "ac": (YA, YC)}.items():
        if x and y:
            vals = [canny_edge_ssim(a, b) for a, b in zip(x, y)]
            deep[f"canny_edge_ssim_{key}"] = round(float(np.mean(vals)), 6)
    if deep.get("canny_edge_ssim_bc") is not None:
        _p(f"  Canny 边缘 SSIM(b,c) = {deep['canny_edge_ssim_bc']}")

    # 3) FID / KID：生成批 vs 真实目标域（testB 斑马）
    zebras = sorted(p for p in SAMPLES.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    and p.name.startswith("zebra"))
    if zebras and YB:
        ref = [np.asarray(load_image(z), dtype=np.uint8) for z in zebras]
        try:
            fa = inception_features(YB)
            fb = inception_features(ref)
            deep["fid"] = {"b_vs_real_zebra": fid(fa, fb),
                           "n_generated": len(YB), "n_real": len(ref)}
            deep["kid"] = kid(fa, fb, n_subsets=20,
                              subset_size=min(50, len(YB), len(ref)))
            _p(f"  FID(B vs 真实斑马) = {deep['fid']['b_vs_real_zebra']}  "
               f"KID = {deep['kid']['mean']}")
        except Exception as e:
            _p(f"  [跳过] FID/KID：{type(e).__name__}: {str(e)[:160]}")
    else:
        _p("  [跳过] FID/KID：samples 里没有 zebra_*.png 作为参照集")

    # 4) 掩码 IoU + 背景 SSIM（真值掩码来自 Mask R-CNN）
    if YA and YC:
        try:
            m_a = instance_masks(YA)
            m_c = instance_masks(YC)
            ious, bgs = [], []
            for row, ma, mc in zip(rows, m_a, m_c):
                if ma["found"] and mc["found"]:
                    ious.append(mask_iou(ma["mask"], mc["mask"]))
                    row["mask_iou_ac"] = round(float(ious[-1]), 6)
                    b = background_ssim(np.asarray(YA[rows.index(row)],
                                                   dtype=np.float32),
                                        np.asarray(YC[rows.index(row)],
                                                   dtype=np.float32),
                                        ma["mask"])
                    bgs.append(b)
                    row["background_ssim_ac"] = round(float(b), 6)
                else:
                    row["mask_iou_ac"] = None
            deep["mask"] = {
                "detected_pairs": len(ious),
                "mean_iou_ac": round(float(np.mean(ious)), 6) if ious else None,
                "mean_background_ssim_ac":
                    round(float(np.mean(bgs)), 6) if bgs else None,
            }
            _p(f"  掩码 IoU(a,c) 均值 = {deep['mask']['mean_iou_ac']}"
               f"（检出 {len(ious)}/{len(rows)} 对）")
        except Exception as e:
            _p(f"  [跳过] 掩码 IoU：{type(e).__name__}: {str(e)[:160]}")

    # 5) 斑马判别准确率（生成结果里有多少被 ImageNet ResNet-50 判成斑马）
    if YC:
        try:
            preds = target_probabilities(YC)
            acc = float(np.mean([p["is_target"] for p in preds]))
            deep["zebra_discrimination"] = {
                "top1_accuracy": round(acc, 6),
                "mean_target_prob": round(
                    float(np.mean([p["target_prob"] for p in preds])), 6),
                "n": len(preds),
            }
            _p(f"  斑马判别 top1 准确率 = {acc:.2%}")
        except Exception as e:
            _p(f"  [跳过] 斑马判别：{type(e).__name__}: {str(e)[:160]}")

    # 6) 异常样本登记：不混进上面的均值；0 例也落盘，便于进度页区分"查过没问题/没查"
    bad = [r for r in rows
           if isinstance(r.get("ssim_bc"), float)
           and r["ssim_bc"] < args.anomaly_threshold]
    if bad:
        records = []
        for r in bad:
            idx = rows.index(r)
            records.append({
                "name": r["sample"],
                "image": YC[idx] if idx < len(YC) else None,
                "reason": f"ssim_bc={r['ssim_bc']} "
                          f"< 阈值 {args.anomaly_threshold}",
                "metrics": {k: v for k, v in r.items() if k != "image"},
            })
        records = [r for r in records if r["image"] is not None]
        if records:
            p = register_anomalies(records, RESULTS / "anomalies")
            deep["anomalies"] = {"count": len(records), "index": str(p),
                                 "threshold": args.anomaly_threshold}
            _p(f"  异常样例 {len(records)} 张 → {p.parent.name}/")
        else:
            deep["anomalies"] = {"count": 0, "index": None,
                                 "threshold": args.anomaly_threshold}
    else:
        deep["anomalies"] = {"count": 0, "index": None,
                             "threshold": args.anomaly_threshold}
        _p(f"  异常样例 0 张（ssim_bc 均 ≥ {args.anomaly_threshold}）")
    return deep

def cmd_layers(args):
    """E4 层间分工：FP32 与 NNCF 模拟 INT8 两侧的逐层表示漂移。"""
    from core.layers import layer_report, load_images
    from core.quantize import collect_calibration

    if not (MODELS / "horse2zebra.pth").exists():
        raise SystemExit("没有预训练权重，先跑 python cli.py setup")

    if not CALIB.exists() or not any(CALIB.iterdir()):
        raise SystemExit("校准集为空，先跑 python cli.py setup")
    calib = collect_calibration(CALIB, args.subset)

    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        exts = {".jpg", ".jpeg", ".png"}
        paths = sorted(p for p in SAMPLES.iterdir()
                       if p.suffix.lower() in exts
                       and p.name.startswith("horse"))[:args.limit]
    if not paths:
        raise SystemExit("没有可用的评测图")

    _p(f"  校准 {len(calib)} 张（trainA），量化侧 NNCF FakeQuantize 模拟")
    images = load_images(paths)
    report = layer_report(MODELS / "horse2zebra.pth", calib, images)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "layers.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _p(f"\n{'层':<16}{'余弦':>10}{'相对L2':>10}{'平均相对误差':>14}")
    for r in report["rows"]:
        _p(f"{r['layer']:<16}{r['cosine']:>10.4f}{r['rel_l2']:>10.4f}"
           f"{r['mean_rel_err']:>14.4f}")
    s = report["summary"]
    _p(f"\n漂移最大层：{s['most_drifted_layer']}（余弦 {s['most_drifted_cosine']}）")
    _p(f"漂移最小层：{s['least_drifted_layer']}（余弦 {s['least_drifted_cosine']}）")
    if s.get("resblock_cosine_head_tail"):
        head, tail = s["resblock_cosine_head_tail"]
        _p(f"残差块首尾余弦 {head:.4f} → {tail:.4f}"
           f"（{'单调下降' if s['drift_monotonic_along_depth'] else '非单调'}）")
    _p(f"已写入 {RESULTS / 'layers.json'}")


# ---------------- serve ----------------

def cmd_serve(args):
    import server
    server.run(host=args.host, port=args.port)


def main():
    ap = argparse.ArgumentParser(description="ShiftLite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="下载模型并完成转换与量化")
    s.add_argument("--force", action="store_true")
    s.add_argument("--subset", type=int, default=CALIB_SUBSET,
                   help=f"校准集张数（方案口径 {CALIB_SUBSET} 张，取 trainA）")
    s.add_argument("--accuracy-control", action="store_true",
                   help="启用精度控制量化（以部分加速收益换质量）")
    s.add_argument("--keep-instance-norm", action="store_true",
                   help="不排除 InstanceNorm/MVN；用于复现量化崩溃现象，勿用于正式产物")
    s.set_defaults(func=cmd_setup)

    b = sub.add_parser("bench", help="A/B/C 三组性能对比")
    # 12 轮计时 + 3 轮预热，丢弃计时首轮（与 README §五的口径一致）
    b.add_argument("--rounds", type=int, default=12)
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--warmup-drop", type=int, default=1)
    b.set_defaults(func=cmd_bench)

    q = sub.add_parser("quality", help="质量与结构保真度量")
    q.add_argument("--limit", type=int, default=4)
    q.add_argument("--deep", action="store_true",
                   help="接入真实感知指标（LPIPS/FID/KID/Canny/掩码/判别），"
                        "首次运行需下载预训练权重")
    q.add_argument("--anomaly-threshold", type=float, default=0.80,
                   help="ssim_bc 低于该值的样本登记为异常样例")
    q.set_defaults(func=cmd_quality)

    ly = sub.add_parser("layers", help="E4 层间分工：逐层表示漂移分析")
    ly.add_argument("--subset", type=int, default=CALIB_SUBSET)
    ly.add_argument("--limit", type=int, default=6,
                    help="参与评测的马图数量（默认用 samples 里的全部样例）")
    ly.add_argument("--images", nargs="+", default=None,
                    help="显式指定评测图，覆盖 --limit")
    ly.set_defaults(func=cmd_layers)

    v = sub.add_parser("serve", help="启动本地可视化界面")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8770)
    v.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    args.func(args)


if __name__ == "__main__":
    main()
