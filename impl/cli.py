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


def download_dataset() -> Path:
    """取样例数据：优先用已有的完整包；否则先下 1.8MB 的 mini（几秒），
    只有在需要 100 张校准集时才值得去下 111MB 的完整包。"""
    # 1) 已有且完整的包直接复用，完整版优先
    for fname, _url, _size in DATASET_SOURCES:
        dst = ART / fname
        if _zip_ok(dst):
            _p(f"  [复用] {fname}（{dst.stat().st_size/1024/1024:.1f} MB）")
            return dst

    # 2) 都没下过：按体积从小到大尝试，避免被大包卡住
    for fname, url, size_mb in sorted(DATASET_SOURCES, key=lambda s: s[2]):
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


def prepare_samples(n_horse: int = 6, n_zebra: int = 3, n_calib: int = 100) -> int:
    """从数据集解压样例图（演示用）与校准集（量化用）。

    校准集必须是 A 域（马）图像，对应方案 §3.3 的「testA 前 100 张」。
    """
    import zipfile

    SAMPLES.mkdir(parents=True, exist_ok=True)
    CALIB.mkdir(parents=True, exist_ok=True)
    zip_path = download_dataset()

    def save(z, name, dst):
        try:
            with z.open(name) as f:
                Image.open(f).convert("RGB").save(dst)
            return True
        except Exception as e:
            _p(f"  [跳过] {name}：{type(e).__name__}")
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
            save(z, name, SAMPLES / f"horse_{i:02d}.png")
            used += 1
        for i, name in enumerate(zebras[:n_zebra], 1):
            save(z, name, SAMPLES / f"zebra_{i:02d}.png")
            used += 1

        # 校准集取 testA 前 n_calib 张，与方案 §3.3 一致
        calib_src = horse_test or horse_train
        calib_n = min(len(calib_src), n_calib)
        for i, name in enumerate(calib_src[:calib_n], 1):
            save(z, name, CALIB / f"calib_{i:03d}.png")

    _p(f"  来源 {zip_path.name}：样例 {used} 张，校准集 {calib_n} 张")
    if calib_n < n_calib:
        _p(f"  注意：校准集只有 {calib_n} 张（方案要求 {n_calib} 张），"
           f"下到完整版 horse2zebra.zip 后重跑 setup --force 可补足。")
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
        _, info = quantize(xml, calib, int8, args.accuracy_control)
        (MODELS / "quantize_report.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        _p(f"  [完成] {int8.name}（校准 {len(calib)} 张）")

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
    s.add_argument("--subset", type=int, default=100)
    s.add_argument("--accuracy-control", action="store_true")
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
