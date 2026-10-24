"""轻迁 CycleLite 命令行入口。

    python cli.py setup      # 下载模型 → 转 ONNX → 转 IR → INT8 量化 → 拉样例图
    python cli.py bench      # A/B/C 三组性能对比（延迟 / FPS / 体积）
    python cli.py quality    # 三组输出的质量与结构保真度量
    python cli.py serve      # 启动本地可视化界面
"""

from __future__ import annotations

import argparse
import json
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

MODEL_URL = ("http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/"
             "horse2zebra.pth")
MODEL_BYTES = 45575747

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
    _resume_download(MODEL_URL, dst, ok)
    _p(f"  [完成] {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
    return dst


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
    _p("== 1/5 预训练权重 ==")
    weights = download_model(force=args.force)

    _p("== 2/5 样例图与校准集 ==")
    n = prepare_samples(n_calib=args.subset)
    _p(f"  样例 {n} 张")

    from core.convert import export_onnx, to_openvino, verify_against_torch

    _p("== 3/5 PyTorch → ONNX ==")
    onnx_path = MODELS / "horse2zebra.onnx"
    if onnx_path.exists() and not args.force:
        _p(f"  [跳过] {onnx_path.name}")
    else:
        export_onnx(weights, onnx_path, args.opset)
        _p(f"  [完成] {onnx_path.name} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")
    _p("== 4/5 ONNX → OpenVINO IR ==")
    xml = MODELS / "horse2zebra.xml"
    if xml.exists() and not args.force:
        _p(f"  [跳过] {xml.name}")
    else:
        to_openvino(onnx_path, xml)
        _p(f"  [完成] {xml.name}")

    report = verify_against_torch(weights, xml)
    _p(f"  转换正确性：max|Δ|={report['max_abs']}  mean|Δ|={report['mean_abs']}")
    (MODELS / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _p("== 5/5 NNCF INT8 量化 ==")
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

    rng = np.random.default_rng(20261014)
    x = rng.standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)

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
            "rounds": int(arr.size),
            "size_mb": rt.model_size_mb(gid),
            "precision": rt.precision(gid),
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
            "input": "1×3×256×256 float32",
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

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(rows),
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
    (RESULTS / "quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"\n已写入 {RESULTS / 'quality.json'}")


# ---------------- serve ----------------

def cmd_serve(args):
    import server
    server.run(host=args.host, port=args.port)


def main():
    ap = argparse.ArgumentParser(description="轻迁 CycleLite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="下载模型并完成转换与量化")
    s.add_argument("--force", action="store_true")
    s.add_argument("--opset", type=int, default=13)
    s.add_argument("--accuracy-control", action="store_true")
    s.add_argument("--subset", type=int, default=CALIB_SUBSET,
                   help=f"校准集张数（方案口径 {CALIB_SUBSET} 张，取 trainA）")
    s.add_argument("--keep-instance-norm", action="store_true",
                   help="不排除 InstanceNorm/MVN；用于复现量化崩溃现象，勿用于正式产物")
    s.set_defaults(func=cmd_setup)

    b = sub.add_parser("bench", help="A/B/C 三组性能对比")
    b.add_argument("--rounds", type=int, default=10)
    b.add_argument("--warmup", type=int, default=2)
    b.add_argument("--warmup-drop", type=int, default=1)
    b.set_defaults(func=cmd_bench)

    q = sub.add_parser("quality", help="质量与结构保真度量")
    q.add_argument("--limit", type=int, default=4)
    q.set_defaults(func=cmd_quality)

    v = sub.add_parser("serve", help="启动本地可视化界面")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8770)
    v.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    args.func(args)


if __name__ == "__main__":
    main()
