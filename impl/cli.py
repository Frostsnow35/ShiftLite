"""ShiftLite 命令行入口。

    python cli.py setup      # 下载模型 → 直转 IR → INT8 量化 → 拉样例图
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
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
MODELS = ART / "models"
SAMPLES = ART / "samples"
RESULTS = ART / "results"

# 权重下载源由 pairs/<id>.json 的 checkpoint 段决定（见下方 model_sources()）。


def model_sources() -> list[tuple[str, str]]:
    """当前 pair 的权重下载源。

    仍只 anchor 一个源：伯克利直链 200 OK、content-length 与 manifest 里的
    bytes 完全对得上，是验过的；HuggingFace 上没有能确认可用的同名权重仓库，
    编一个 URL 进去只会让人白等。要用备源就用环境变量
    SHIFTLITE_WEIGHTS_URL 指过去，它会被排在最前面。
    """
    src = [(f"{PAIR.id} 官方", PAIR.ckpt_url)]
    override = os.environ.get("SHIFTLITE_WEIGHTS_URL", "").strip()
    if override:
        return [("环境变量 SHIFTLITE_WEIGHTS_URL", override)] + src
    return src

# ── 迁移对（pair）：把「换权重即换应用」从代码挪到配置 ──────────────
# 权威配置在 pairs/<id>.json。默认 pair 沿用既有平铺路径（已入库的正式口径
# 证据不受影响），其余 pair 落在 artifacts/<id>/ 下，互不覆盖。
from core.pairs import DEFAULT_PAIR, list_pairs, load_pair  # noqa: E402

PAIR = load_pair(DEFAULT_PAIR)
MODELS = PAIR.paths.models
SAMPLES = PAIR.paths.samples
CALIB = PAIR.paths.calib
RESULTS = PAIR.paths.results
EVAL = ART / "eval"
EVAL_IN = PAIR.paths.eval_in
EVAL_REF = PAIR.paths.eval_ref
MODEL_BYTES = PAIR.ckpt_bytes


def _dataset_sources(pair) -> list[tuple[str, str, float]]:
    """当前 pair 的主数据集包 + mini 兜底。

    mini 每个域只有 10 张（实测 45 个条目），只能用来跑通流程，
    支撑不了「校准 trainA 300 张 / 评测 testA+testB 全量」的正式口径。
    """
    return [
        (pair.dataset_zip, pair.dataset_url, pair.dataset_mb),
        ("mini.zip", "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/"
                     "mini.zip", 1.8),
    ]


DATASET_SOURCES = _dataset_sources(PAIR)

# 方案 §3.3 口径：校准集取 trainA 前 300 张，与评测所用 test 集隔离；
# 演示/评测样例取 testA、testB。三者来源不同，不可互相替代。
CALIB_SUBSET = 300
N_SAMPLE_HORSE = 6
N_SAMPLE_ZEBRA = 3


def select_pair(pair_id: str = DEFAULT_PAIR) -> None:
    """切换当前迁移对：重绑全部路径与数据来源常量。

    用模块级全局是为了让既有函数签名保持不动（它们都直接读 MODELS/SAMPLES/…）。
    CLI 是单次进程、子命令互斥，这么做不会有并发问题。
    """
    global PAIR, MODELS, SAMPLES, CALIB, RESULTS, EVAL_IN, EVAL_REF
    global MODEL_BYTES, DATASET_SOURCES
    p = load_pair(pair_id)
    PAIR = p
    MODELS = p.paths.models
    SAMPLES = p.paths.samples
    CALIB = p.paths.calib
    RESULTS = p.paths.results
    EVAL_IN = p.paths.eval_in
    EVAL_REF = p.paths.eval_ref
    MODEL_BYTES = p.ckpt_bytes
    DATASET_SOURCES = _dataset_sources(p)

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


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


def _content_length(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def _parallel_download(url: str, dst: Path, total: int, parts: int = 4) -> Path:
    """分段并发下载（每段写独立 .partN，最后拼接）。

    为什么需要它：伯克利那台 Apache 对**长时间单连接**限速很狠。实测单连接
    跑满 45 MB 平均只有 ~80 KB/s，而每次只请求 4 MB 的有界区间能到 ~180 KB/s。
    分段后每段新建连接，既绕开长连接限速，也避免"卡住要等 90 秒超时"的空转。

    为什么不预分配目标文件再各段 seek 写入：那样 `stat` 会立刻显示文件已达
    完整大小，看起来像"秒下完"，实际分片还在写——进度不可观测，也容易被误判
    成成功。写独立分片文件可以让「已下载字节数」= 各分片大小之和，真实可信。

    注意：**同一个目标文件绝不能被两个进程同时下载**——两边各自按"已有大小"
    推算 Range 起点，会互相覆盖成坏文件。
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    dst.parent.mkdir(parents=True, exist_ok=True)
    seg = (total + parts - 1) // parts
    ranges = [(i * seg, min(total, (i + 1) * seg) - 1) for i in range(parts)]
    part_paths = [dst.with_name(f"{dst.name}.part{i}") for i in range(parts)]

    def worker(i: int) -> None:
        start, end = ranges[i]
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={start}-{end}")
        with urllib.request.urlopen(req, timeout=180) as r, open(part_paths[i], "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)

    try:
        with ThreadPoolExecutor(max_workers=parts) as ex:
            for fu in [ex.submit(worker, i) for i in range(parts)]:
                fu.result()

        with open(dst, "wb") as out:
            for i, (start, end) in enumerate(ranges):
                expect = end - start + 1
                got = part_paths[i].stat().st_size
                if got != expect:
                    raise RuntimeError(f"分片 {i} 大小不符：{got}/{expect}")
                with open(part_paths[i], "rb") as f:
                    shutil.copyfileobj(f, out)
    finally:
        for p in part_paths:
            p.unlink(missing_ok=True)

    return dst


def _ensure_file(url: str, dst: Path, total: int,
                 validator=None, parts: int = 4) -> Path:
    """把 dst 备齐到 total 字节：能用就复用，不能用就分段并发重下。"""
    ok = validator or (lambda p: p.exists() and p.stat().st_size >= total)
    if ok(dst):
        return dst
    if dst.exists():
        _p(f"  文件不完整（{dst.stat().st_size}/{total} B），重新分段下载…")
        dst.unlink()
    _p(f"  分段并发下载（{parts} 路，{total/1024/1024:.1f} MB）…")
    try:
        _parallel_download(url, dst, total, parts)
    except Exception as e:
        _p(f"  分段下载失败（{type(e).__name__}），回退单连接续传…")
        if dst.exists():
            dst.unlink()
        _resume_download(url, dst, validator)
        return dst
    if not ok(dst):
        raise SystemExit(f"下载校验失败：{dst}（{dst.stat().st_size}/{total} B）")
    return dst


def download_model(force: bool = False) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    dst = PAIR.weights
    ok = lambda p: p.exists() and p.stat().st_size >= MODEL_BYTES
    if ok(dst) and not force:
        _p(f"  [跳过] 权重已存在 {dst.stat().st_size/1024/1024:.1f} MB")
        return dst
    _p("  下载预训练权重（分段并发，支持断点续传）…")
    for label, url in model_sources():
        try:
            _p(f"  尝试 {label}…")
            _ensure_file(url, dst, MODEL_BYTES, ok)
            if ok(dst):
                _p(f"  [完成] {dst} ({dst.stat().st_size/1024/1024:.1f} MB)"
                   f"，来源：{label}")
                return dst
            _p(f"  [放弃] {label} 下载不完整，换下一个源")
        except SystemExit:
            _p(f"  [放弃] {label} 不可用，换下一个源")
            continue
    raise SystemExit(
        f"权重下载失败：所有镜像都不可用。手动下载 {PAIR.ckpt_file} 放到 "
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
            total = _content_length(url)
            if total:
                _p(f"  下载 {fname}（{total/1024/1024:.1f} MB，分段并发）…")
                _ensure_file(url, dst, total, _zip_ok)
            else:
                _p(f"  下载 {fname}（约 {size_mb} MB，单连接续传）…")
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

    def _sel(names: list[str], sub: str) -> list[str]:
        return sorted(n for n in names
                      if f"/{sub}/" in n and n.lower().endswith(".jpg"))

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # 样例优先取官方测试集；测试集缺失时才退回训练集
        inputs = (_sel(names, PAIR.input_dir)
                  or _sel(names, PAIR.input_dir.replace("test", "train")))
        refs = (_sel(names, PAIR.ref_dir)
                or _sel(names, PAIR.ref_dir.replace("test", "train")))
        if not inputs:
            raise SystemExit(
                f"{zip_path.name} 里找不到输入域图像（{PAIR.input_dir}）：{PAIR.label}")

        used = 0
        for i, name in enumerate(inputs[:n_horse], 1):
            if save(z, name, SAMPLES / f"{PAIR.input_prefix}_{i:02d}.png"):
                used += 1
        for i, name in enumerate(refs[:n_zebra], 1):
            if save(z, name, SAMPLES / f"{PAIR.ref_prefix}_{i:02d}.png"):
                used += 1

        # 校准集：trainA 前 n_calib 张（方案 §3.3），与 testA/testB 评测集隔离。
        calib_src = _sel(names, PAIR.calib_dir)
        if not calib_src:
            raise SystemExit(
                f"{zip_path.name} 里没有 {PAIR.calib_dir}（训练集）图像，无法准备校准集。\n"
                f"        请删除 {zip_path} 后重跑 setup 以下载完整版 {PAIR.dataset_zip}。")
        calib_n = min(len(calib_src), n_calib)
        written = []
        for i, name in enumerate(calib_src[:calib_n], 1):
            if save(z, name, CALIB / f"calib_{i:03d}.png"):
                written.append(name)

        # 完整评测集（方案 §4）：输入域/参照域全量，与 samples/ 演示样例分开存放。
        n_in, n_ref = _extract_eval_sets(z, skipped, zip_path.name)

    # 留一份来源清单，报告里可以据此证明校准集与评测集不重叠
    (CALIB / "manifest.json").write_text(
        json.dumps({
            "source_zip": zip_path.name,
            "calib_prefix": dir_prefix(calib_src),
            "requested": n_calib,
            "written": len(written),
            "eval_samples_prefix": dir_prefix(inputs[:n_horse]),
            "files": written[:10],
            "skipped_count": len(skipped),
            "skipped": skipped[:20],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    _p(f"  来源 {zip_path.name}：样例 {used} 张，校准集 {len(written)} 张，"
       f"评测集 输入批 {n_in} 张（{PAIR.input_dir}）/ 参照批 {n_ref} 张（{PAIR.ref_dir}）")
    if skipped:
        _p(f"  ⚠️ {len(skipped)} 个 zip 条目解压失败已跳过（见 calib/manifest.json）")
    if len(written) < n_calib:
        _p(f"  ⚠️ 校准集只有 {len(written)} 张，方案 §3.3 要求 trainA 前 {n_calib} 张；"
           f"请在报告中如实写明实际张数，或重跑 setup 取完整数据集。")
    return used


def _extract_eval_sets(z, skipped: list[str], zip_name: str) -> tuple[int, int]:
    """从已打开的数据集 zip 解压完整评测集（方案 §4）。

    按「角色」落盘，不按 zip 的 testA/testB 落盘：
      输入域（pair.input_dir）→ eval/input，参照域（pair.ref_dir）→ eval/ref。
    原始 jpg 字节直接落盘（不重编码），与 zip 条目逐一对应。
    """
    names = z.namelist()
    inp = sorted(n for n in names
                 if f"/{PAIR.input_dir}/" in n and n.lower().endswith(".jpg"))
    ref = sorted(n for n in names
                 if f"/{PAIR.ref_dir}/" in n and n.lower().endswith(".jpg"))
    if not inp and not ref:
        return 0, 0

    EVAL_IN.mkdir(parents=True, exist_ok=True)
    EVAL_REF.mkdir(parents=True, exist_ok=True)
    n_a = n_b = 0
    for name in inp:
        try:
            (EVAL_IN / Path(name).name).write_bytes(z.read(name))
            n_a += 1
        except Exception as e:
            skipped.append(f"{name}（{type(e).__name__}）")
    for name in ref:
        try:
            (EVAL_REF / Path(name).name).write_bytes(z.read(name))
            n_b += 1
        except Exception as e:
            skipped.append(f"{name}（{type(e).__name__}）")

    (EVAL_IN.parent / "manifest.json").write_text(
        json.dumps({
            "source_zip": zip_name,
            "pair": PAIR.id,
            "input": {"role": "生成批", "from": PAIR.input_dir,
                      "entries": len(inp), "written": n_a},
            "ref": {"role": "FID/KID 参照批", "from": PAIR.ref_dir,
                    "entries": len(ref), "written": n_b},
            "protocol": "方案 §4：生成批 = 输入域全量，FID/KID 参照批 = 目标域全量",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_a, n_b


def prepare_eval() -> tuple[int, int]:
    """确保完整评测集就位（eval/input、eval/ref），返回 (输入批数, 参照批数)。

    已解压则直接返回；本地没有数据集 zip 时安静跳过，
    调用方回退到 samples/ 的演示样例并提示样本量不足。
    """
    def count(d: Path) -> list[Path]:
        return ([p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS]
                if d.exists() else [])

    a, b = count(EVAL_IN), count(EVAL_REF)
    if a and b:
        return len(a), len(b)

    zip_path = next((ART / name for name, _, _ in DATASET_SOURCES
                     if (ART / name).exists()), None)
    if zip_path is None:
        return 0, 0

    skipped: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        return _extract_eval_sets(z, skipped, zip_path.name)


def _list_imgs(d: Path, prefix: str | None = None) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in _IMG_EXTS
                  and (prefix is None or p.name.startswith(prefix)))


def eval_input_paths(limit: int | None = None) -> list[Path]:
    """生成批输入：优先 eval/input 全量（方案 §4），退而求其次 samples/<input_prefix>_*。"""
    ps = _list_imgs(EVAL_IN) or _list_imgs(SAMPLES, PAIR.input_prefix)
    return ps[:limit] if limit else ps


def eval_ref_paths(limit: int | None = None) -> list[Path]:
    """FID/KID 参照批（真实目标域）：优先 eval/ref 全量，退而其次 samples/<ref_prefix>_*。"""
    ps = _list_imgs(EVAL_REF) or _list_imgs(SAMPLES, PAIR.ref_prefix)
    return ps[:limit] if limit else ps


# 兼容旧名：既有调用点与报告口径按 horse/zebra 命名，语义已由 pair 决定
def eval_horse_paths(limit: int | None = None) -> list[Path]:
    return eval_input_paths(limit)


def eval_zebra_paths(limit: int | None = None) -> list[Path]:
    return eval_ref_paths(limit)


def cmd_setup(args):
    _p("== 1/4 预训练权重 ==")
    weights = download_model(force=args.force)

    _p("== 2/4 样例图与校准集 ==")
    n = prepare_samples(n_calib=args.subset)
    _p(f"  样例 {n} 张")

    from core.convert import build_ir, verify_against_torch

    _p("== 3/4 PyTorch 直转 OpenVINO IR（不经 ONNX）==")
    xml = PAIR.xml
    if xml.exists() and not args.force:
        _p(f"  [跳过] {xml.name}")
    else:
        build_ir(weights, xml)
        _p(f"  [完成] {xml.name}")

    # 判据要用真实图像：造出来的随机输入值域约 ±3，不是真实输入分布
    ref = sorted(SAMPLES.glob(f"{PAIR.input_prefix}_*.png"))
    report = verify_against_torch(weights, xml, image=ref[0] if ref else None)
    _p(f"  验证输入：{report.get('input_image') or '固定随机输入（样例图缺失）'}")
    _p(f"  转换正确性：max|Δ|={report['max_gray']} 灰度级，"
       f"mean|Δ|={report['mean_gray']} 灰度级"
       f"（判据 {'通过' if report['pass'] else '未通过'}：均值≤0.5、最大≤5）")
    _p(f"  IR 规模：{report['total_ops']} 个节点 / {report['op_kinds']} 类算子")
    (MODELS / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _p("== 4/4 NNCF INT8 量化 ==")
    int8 = PAIR.int8_xml
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
    from core.mem import rss_mb
    from core.utils import SIZE

    rt = Runtime(PAIR.weights, PAIR.xml, PAIR.int8_xml)
    avail = rt.availability()
    if not any(avail.values()):
        raise SystemExit("没有任何可用模型，先跑 python cli.py setup")

    # 测速输入也换成真实图像：与时延相关的是计算量，用真实分布能顺带记录
    # 一次有代表性的输入；取不到样例图时退回固定随机输入。
    ref = sorted(SAMPLES.glob(f"{PAIR.input_prefix}_*.png"))
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

        # 内存口径：进程 RSS。采样放在计时窗口**之外**，不污染延迟测量。
        rss_before = rss_mb()
        times = []
        rss_peak = rss_before
        for i in range(args.rounds):
            t0 = time.perf_counter()
            rt.infer(x, gid)
            times.append((time.perf_counter() - t0) * 1000)
            r = rss_mb()
            if r is not None and (rss_peak is None or r > rss_peak):
                rss_peak = r
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
            # 内存四项里的第四项（PLAN §4）。三组在同一进程里顺序加载，
            # 后测的组 RSS 绝对值含先前运行时的常驻，跨组横向比较要看
            # rss_delta_mb（推理期间峰值 − 本组推理前基线）。
            "rss_before_mb": None if rss_before is None else round(rss_before, 1),
            "rss_peak_mb": None if rss_peak is None else round(rss_peak, 1),
            "rss_delta_mb": (None if (rss_before is None or rss_peak is None)
                             else round(rss_peak - rss_before, 1)),
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
            # 内存：进程 RSS(MB)。rss_before=本组推理前（模型已加载）基线，
            # rss_peak=推理期间采样峰值，rss_delta=两者之差。三组分先后在同一
            # 进程加载，绝对值不可横向比，增量可比。
            "memory_metric": "进程 RSS (MB)：rss_peak − rss_before",
        },
        "groups": groups,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    _p("\n组别  均值(ms)  标准差  FPS    体积(MB)  相对A加速  RSS峰值(MB)  推理增量(MB)")
    for g in groups:
        if g.get("available"):
            _p(f"{g['id']}     {g['mean_ms']:>7.1f}  {g['std_ms']:>6.1f}  "
               f"{g['fps']:>6.1f}  {g['size_mb']:>7.2f}   {g.get('speedup_vs_A','-')}x"
               f"       {g.get('rss_peak_mb','-')}        {g.get('rss_delta_mb','-')}")


# ---------------- quality ----------------

def cmd_quality(args):
    from core.backends import Runtime
    from core.metrics import (ssim, edge_ssim, mae, psnr, lpips_three_way)
    from core.utils import load_image, to_tensor, to_image

    rt = Runtime(PAIR.weights, PAIR.xml, PAIR.int8_xml)
    avail = rt.availability()

    out_dir = RESULTS / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_eval()
    horses = eval_horse_paths(args.limit)
    if not horses:
        raise SystemExit("没有评测马图，先跑 python cli.py setup")
    if len(horses) < args.limit:
        _p(f"  ⚠️ 评测图只有 {len(horses)} 张（--limit {args.limit}），"
           f"方案 §4 口径是输入域全量（{PAIR.input_dir} 全部），报告请如实写明实际样本量")

    rows = []
    YA, YB, YC, inputs = [], [], [], []
    for p in horses:
        img = load_image(p)
        inputs.append(np.asarray(img, dtype=np.uint8))
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

        row = {"sample": p.stem, "input": p.relative_to(ROOT).as_posix()}
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
        deep_block = _deep_metrics(YA, YB, YC, rows, out_dir, args, inputs)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": "quality-v2",
        "n_samples": len(rows),
        "metrics_source": ("real" if deep_block.get('complete') else "partial") if deep_block else "proxy",
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


def _deep_metrics(YA, YB, YC, rows, out_dir, args, inputs) -> dict:
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
    deep: dict = {"requested": True, "groups": have, "errors": [],
                  "fid_implementation": "torchvision ImageNet InceptionV3 legacy; not standard pytorch-fid comparable",
                  "classifier_preprocessing": "ResNet50_Weights.IMAGENET1K_V1.transforms()"}

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
    for key, (x, y) in {"ab": (YA, YB), "bc": (YB, YC), "ac": (YA, YC),
                       "xa": (inputs, YA), "xb": (inputs, YB), "xc": (inputs, YC)}.items():
        if x and y:
            vals = [canny_edge_ssim(a, b) for a, b in zip(x, y)]
            deep[f"canny_edge_ssim_{key}"] = round(float(np.mean(vals)), 6)
            for row, value in zip(rows, vals):
                row[f'canny_edge_ssim_{key}'] = value
    deep['canny_n_valid'] = {key: sum(r.get(f'canny_edge_ssim_{key}') is not None for r in rows)
                             for key in ('ab', 'bc', 'ac', 'xa', 'xb', 'xc')}
    if deep.get("canny_edge_ssim_bc") is not None:
        _p(f"  Canny 边缘 SSIM(b,c) = {deep['canny_edge_ssim_bc']}")

    # 3) FID / KID：方案 §4 要求两批生成结果（FP32 批 b、INT8 批 c）
    #    分别与同一批真实目标域（testB 斑马全量）比较，不能只报 B 批。
    zebras = eval_zebra_paths()
    if zebras and (YB or YC):
        ref = [np.asarray(load_image(z), dtype=np.uint8) for z in zebras]
        try:
            fb = inception_features(ref)
            fid_block = {"n_generated": len(YB) or len(YC), "n_real": len(ref),
                         "ref_domain": PAIR.ref_prefix}
            kid_block = {}
            for tag, Y in (("b", YB), ("c", YC)):
                if not Y:
                    continue
                f = inception_features(Y)
                # 键名带参照域前缀（斑马对仍是 *_vs_real_zebra，与原口径一致；
                # 反向对 zebra2horse 的参照域是马，就会写成 *_vs_real_horse，
                # 不会出现「参照的是马、字段却叫 zebra」这种误导）
                field = f"{tag}_vs_real_{PAIR.ref_prefix}"
                fid_block[field] = fid(f, fb)
                kid_block[field] = kid(
                    f, fb, n_subsets=100,
                    subset_size=min(50, len(Y), len(ref)))
                _p(f"  FID({tag} vs 真实目标域 {PAIR.ref_prefix}) = "
                   f"{fid_block[field]}  "
                   f"KID = {kid_block[field]['mean']}"
                   f"（subset {kid_block[field]['subset_size']}）")
            deep["fid"] = fid_block
            deep["kid"] = kid_block
        except Exception as e:
            deep['errors'].append({'metric': 'fid/kid', 'error': str(e)})
            _p(f"  [跳过] FID/KID：{type(e).__name__}: {str(e)[:160]}")
    else:
        _p(f"  [跳过] FID/KID：没有目标域参照批（eval/{PAIR.ref_dir} 或 samples）")

    # 输入保真 X→A/B/C 与输出一致性 A→C 分开；掩码不是人工真值。
    # 域型/风格型 pair 没有稳定前景、也无可对齐的 COCO 类别，按 manifest 关闭。
    if all(have.values()) and PAIR.metric_enabled("mask_iou"):
        try:
            from core.structure import measure_structure, write_review_materials
            images = {'X': inputs, 'A': YA, 'B': YB, 'C': YC}
            detections = {}
            for side, batch in images.items():
                _p(f"  Mask R-CNN: {side} ({len(batch)} images)")
                # 检测类随迁移对方向取：输入域（X）用 input_prefix，输出域用 ref_prefix。
                # 仅 horse2zebra / zebra2horse 启用掩码 IoU，两者前缀恰为 COCO 类。
                label = PAIR.input_prefix if side == 'X' else PAIR.ref_prefix
                detections[side] = instance_masks(batch, label=label)
            structure = measure_structure(rows, images, detections)
            write_review_materials(structure, rows, images, detections, RESULTS / 'structure_review')
            deep['structure'] = structure
            # 旧 A/C 背景填充口径保留为 legacy，严格共同背景在 structure.pairs。
            bgs = []
            for i, row in enumerate(rows):
                da, dc = detections['A'][i], detections['C'][i]
                value = None
                if da['found'] and dc['found']:
                    try:
                        value = background_ssim(YA[i], YC[i], da['mask'])
                    except ValueError:
                        pass
                row['background_ssim_ac'] = value
                if value is not None:
                    bgs.append(value)
            pair = structure['pairs']['a_c']
            deep['mask'] = {
                'detected_pairs': pair['iou']['n_valid'],
                'mean_iou_ac': pair['iou']['mean'], 'mean_dice_ac': pair['dice']['mean'],
                'mean_background_ssim_ac': round(float(np.mean(bgs)), 6) if bgs else None,
                'background_n_valid': len(bgs), 'n_total': len(rows),
                'background_protocol': 'legacy foreground-fill whole-image SSIM',
            }
            _p(f"  结构评测完成：{structure['n_failure_samples']} 张检测失败，人工待复核")
        except Exception as e:
            deep['errors'].append({'metric': 'structure', 'error': str(e)})
            _p(f"  [失败] 结构评测：{type(e).__name__}: {e}")
    elif not PAIR.metric_enabled("mask_iou"):
        _p(f"  [不适用] 结构评测 / 掩码 IoU / 背景 SSIM："
           f"{PAIR.metric_reason('mask_iou') or '该 pair 未启用'}")

    # 5) 目标类别判别（任务保真；方案 §4「量化前后」→ b、c 两批各评一次）
    #    ImageNet 没有 winter / 画作等类别，域型与风格型 pair 按 manifest 关闭
    disc = {}
    for tag, Y in ((("b", YB), ("c", YC))
                   if PAIR.metric_enabled("class_discrimination") else ()):
        if not Y:
            continue
        try:
            preds = target_probabilities(Y, substrings=(PAIR.classifier_label(),))
            acc = float(np.mean([p["is_target"] for p in preds]))
            disc[tag] = {
                "top1_accuracy": round(acc, 6),
                "mean_target_prob": round(
                    float(np.mean([p["target_prob"] for p in preds])), 6),
                "n": len(preds),
            }
            _p(f"  目标类别判别 top1（{tag} 批，{PAIR.ref_prefix}）= "
               f"{acc:.2%}（n={len(preds)}）")
        except Exception as e:
            _p(f"  [跳过] 目标类别判别（{tag} 批）：{type(e).__name__}: {str(e)[:160]}")
            deep['errors'].append({'metric': f'classifier_{tag}', 'error': str(e)})
    if disc:
        # 键名不再写死 zebra：反向对的目标类就是 horse。
        # 旧键名 zebra_discrimination 仍在界面侧兼容读取，避免已入库 JSON 失效。
        deep["class_discrimination"] = {"target": PAIR.ref_prefix, "by_batch": disc}

    # 异常仍计入全集均值；零异常也写空索引，避免残留旧登记。
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
            # 落盘相对路径（相对 impl/）：产物 JSON 要进仓库，写本机绝对路径
            # 会让队友 clone 后该字段直接失效。读取侧按脚本根拼接。
            deep["anomalies"] = {"count": len(records),
                                 "index": str(p.relative_to(ROOT)),
                                 "threshold": args.anomaly_threshold}
            _p(f"  异常样例 {len(records)} 张 → {p.parent.name}/")
        else:
            deep["anomalies"] = {"count": 0, "index": None,
                                 "threshold": args.anomaly_threshold}
    else:
        register_anomalies([], RESULTS / 'anomalies')
        deep["anomalies"] = {"count": 0, "index": None,
                             "threshold": args.anomaly_threshold}
        _p(f"  异常样例 0 张（ssim_bc 均 ≥ {args.anomaly_threshold}）")
    # 必需项按 manifest 指标开关动态构造：关闭的 pair 不应恒为 partial。
    required = ['lpips', 'fid', 'kid']
    not_applicable = []
    if PAIR.metric_enabled('mask_iou'):
        required += ['structure', 'mask']
    else:
        not_applicable.append(
            f"掩码 IoU / 结构评测 / 背景 SSIM：{PAIR.metric_reason('mask_iou') or '该 pair 未启用'}")
    if PAIR.metric_enabled('class_discrimination'):
        required += ['class_discrimination']
    else:
        not_applicable.append(
            f"类别判别：{PAIR.metric_reason('class_discrimination') or '该 pair 未启用'}")
    disc_ok = (set(disc) == {'b', 'c'}) if PAIR.metric_enabled('class_discrimination') else True
    deep['not_applicable'] = not_applicable
    deep['complete'] = (all(have.values()) and not deep['errors']
                        and all(k in deep for k in required)
                        and disc_ok)
    return deep

def cmd_layers(args):
    """E4 层间分工：FP32 与 NNCF 模拟 INT8 两侧的逐层表示漂移。"""
    from core.layers import layer_report, load_images
    from core.quantize import collect_calibration

    if not PAIR.weights.exists():
        raise SystemExit("没有预训练权重，先跑 python cli.py setup")

    if not CALIB.exists() or not any(CALIB.iterdir()):
        raise SystemExit("校准集为空，先跑 python cli.py setup")
    calib = collect_calibration(CALIB, args.subset)

    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        prepare_eval()
        paths = eval_horse_paths(args.limit)
    if not paths:
        raise SystemExit("没有可用的评测图")
    if not args.images and len(paths) < args.limit:
        _p(f"  ⚠️ 评测图只有 {len(paths)} 张（--limit {args.limit}），"
           f"方案 §4 口径是 testA 全量 120 张，报告请如实写明实际样本量")

    _p(f"  校准 {len(calib)} 张（{PAIR.calib_dir}），量化侧 NNCF FakeQuantize 模拟")
    images = load_images(paths)
    report = layer_report(PAIR.weights, calib, images)

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


# ---------------- figures ----------------

def cmd_figures(args):
    from core.figures import make_all
    produced = make_all(pair=PAIR)
    _p(f"实验图表已生成（{PAIR.id}，issue #14）：")
    for p in produced:
        _p(f"  {p}")


def cmd_stylize(args):
    """把参数原样转给 stylize/run.py。

    支线自带一套子命令（poc / convert / quantize / bench / download），
    这里只做入口转发，避免把 AdaIN 的实现细节塞进主线 cli.py。
    """
    import runpy
    target = ROOT / "stylize" / "run.py"
    if not target.exists():
        raise SystemExit(f"找不到 AdaIN 支线入口：{target}")
    sys.argv = [str(target)] + list(args.stylize_args)
    runpy.run_path(str(target), run_name="__main__")


# ---------------- review materials ----------------

def cmd_review_materials(args):
    from core.review import build_review_materials
    manifest = build_review_materials(args.quality, args.output, args.image_root,
                                      args.sample_list)
    _p(f"已整理 {len(manifest['samples'])} 张待复核材料 → {args.output}")
    _p("人工填写请另存 human_review.csv；命令不会填写或覆盖人工结论。")


# ---------------- serve ----------------

def cmd_serve(args):
    import server
    server.run(host=args.host, port=args.port)


def main():
    ap = argparse.ArgumentParser(description="ShiftLite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 所有子命令共用 --pair：迁移对由 pairs/<id>.json 决定，不改代码即可换域
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pair", default=DEFAULT_PAIR, choices=list_pairs(),
                        help="迁移对（pairs/<id>.json）；默认 horse2zebra，沿用既有平铺路径")

    s = sub.add_parser("setup", parents=[common], help="下载模型并完成转换与量化")
    s.add_argument("--force", action="store_true")
    s.add_argument("--subset", type=int, default=CALIB_SUBSET,
                   help=f"校准集张数（方案口径 {CALIB_SUBSET} 张，取 trainA/trainB）")
    s.add_argument("--accuracy-control", action="store_true",
                   help="启用精度控制量化（以部分加速收益换质量）")
    s.add_argument("--keep-instance-norm", action="store_true",
                   help="不排除 InstanceNorm/MVN；用于复现量化崩溃现象，勿用于正式产物")
    s.set_defaults(func=cmd_setup)

    b = sub.add_parser("bench", parents=[common], help="A/B/C 三组性能对比")
    # 12 轮计时 + 3 轮预热，丢弃计时首轮（与 README §五的口径一致）
    b.add_argument("--rounds", type=int, default=12)
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--warmup-drop", type=int, default=1)
    b.set_defaults(func=cmd_bench)

    q = sub.add_parser("quality", parents=[common], help="质量与结构保真度量")
    q.add_argument("--limit", type=int, default=4,
                   help="参与评测的输入图数量（方案 §4 正式口径为输入域全量）")
    q.add_argument("--deep", action="store_true",
                   help="接入真实感知指标（LPIPS/FID/KID/Canny/掩码/判别），"
                        "首次运行需下载预训练权重")
    q.add_argument("--anomaly-threshold", type=float, default=0.80,
                   help="ssim_bc 低于该值的样本登记为异常样例")
    q.set_defaults(func=cmd_quality)

    ly = sub.add_parser("layers", parents=[common],
                        help="E4 层间分工：逐层表示漂移分析")
    ly.add_argument("--subset", type=int, default=CALIB_SUBSET)
    ly.add_argument("--limit", type=int, default=6,
                    help="参与评测的输入图数量（取 eval/输入域；方案 §4 正式口径为全量）")
    ly.add_argument("--images", nargs="+", default=None,
                    help="显式指定评测图，覆盖 --limit")
    ly.set_defaults(func=cmd_layers)

    v = sub.add_parser("serve", parents=[common], help="启动本地可视化界面")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8770)
    v.set_defaults(func=cmd_serve)

    fg = sub.add_parser("figures", parents=[common],
                        help="生成实验图表（E2 柱状图/E3 异常拼图/E4 层间曲线）")
    fg.set_defaults(func=cmd_figures)

    # AdaIN 支线（issue #32）：独立命令空间，明细见 stylize/README.md。
    # 与 A/B/C 主线口径隔离：产物落 artifacts/adain/，不写 artifacts/results/。
    st = sub.add_parser(
        "stylize",
        help="AdaIN 支线：任意参考图风格化（独立的 poc/convert/quantize/bench）")
    st.add_argument("stylize_args", nargs=argparse.REMAINDER,
                    help="透传给 stylize/run.py 的参数，例如 "
                         "`poc --content a.jpg --style b.jpg --alpha 0.8`")
    st.set_defaults(func=cmd_stylize)

    rv = sub.add_parser("review-materials", help="从已有 quality-v2 结果整理复核图片、清单和空白 CSV，不重跑模型")
    rv.add_argument("--quality", type=Path, default=RESULTS / 'quality.json')
    rv.add_argument("--output", type=Path, required=True, help="独立材料目录；不同实验请用新目录")
    rv.add_argument("--image-root", type=Path, default=ROOT, help="JSON 中 input 路径的基准目录（impl）")
    rv.add_argument("--sample-list", type=Path, help="可选 JSON：{samples: [样本ID, ...]}，按给定顺序导出")
    rv.set_defaults(func=cmd_review_materials)

    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    # 迁移对绑定：只有声明了 --pair 的子命令才需要重绑路径。
    # stylize（AdaIN 支线）走的是另一套入口，没有 --pair，直接 getattr 会崩。
    if hasattr(args, "pair"):
        select_pair(args.pair)      # 先按 --pair 重绑路径，再执行子命令
    args.func(args)


if __name__ == "__main__":
    main()
