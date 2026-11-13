"""AdaIN 支线命令行入口（issue #32）。

用法（在 impl/ 目录下执行）：
    python stylize/run.py download                 # 拉官方权重（约 94 MB）
    python stylize/run.py poc                      # 内容图 + 风格图 → 出图
    python stylize/run.py convert                  # PyTorch → OpenVINO IR + 一致性验证
    python stylize/run.py quantize --exclude-norm  # NNCF PTQ（含"排除归一化"对照）
    python stylize/run.py bench                    # A/B/C 三组延迟

隔离约定：产物只落 `stylize/weights/` 与 `artifacts/adain/`，
不碰 `artifacts/results/`（那是主线 A/B/C 的证据目录）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # impl/stylize
IMPL = ROOT.parent                              # impl
sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(ROOT))

ART = IMPL / "artifacts" / "adain"
OUT = ART / "out"
IR = ART / "ir"
WEIGHTS = ROOT / "weights"
SAMPLES = ROOT / "samples"

RELEASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/"
FILES = (("decoder.pth", 14023458), ("vgg_normalised.pth", 80102481))

IR_SIZE = 256          # IR 固定输入边长（OpenVINO 静态形状）
A1 = __import__("numpy").array([1.0], dtype="float32")   # alpha 默认满强度


def _p(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------- 下载 ----------------

def _range_worker(url: str, dst: Path, start: int, end: int) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "ShiftLite-AdaIN/1.0"})
    req.add_header("Range", f"bytes={start}-{end}")
    got = 0
    with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
    return got


def _fetch(url: str, dst: Path, total: int, parts: int = 4) -> Path:
    """4 段并发。实测单连接会被限速、卡顿还要等超时，分段能稳住在 100–200 KB/s。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    seg = (total + parts - 1) // parts
    jobs = [(i * seg, min(total, (i + 1) * seg) - 1) for i in range(parts)]
    part_paths = [dst.with_name(dst.name + f".p{i}") for i in range(parts)]
    with ThreadPoolExecutor(max_workers=parts) as ex:
        futs = [ex.submit(_range_worker, url, pp, s, e)
                for pp, (s, e) in zip(part_paths, jobs)]
        for f in futs:
            f.result()
    with open(dst, "wb") as out:
        for pp in part_paths:
            out.write(pp.read_bytes())
            pp.unlink()
    if dst.stat().st_size < total:
        raise SystemExit(f"{dst.name} 不完整：{dst.stat().st_size}/{total}")
    return dst


def cmd_download(args):
    for name, size in FILES:
        dst = WEIGHTS / name
        if dst.exists() and dst.stat().st_size == size:
            _p(f"  [跳过] {name} 已存在（{size/1024/1024:.1f} MB）")
            continue
        _p(f"  下载 {name}（{size/1024/1024:.1f} MB，4 段并发）…")
        t0 = time.time()
        _fetch(RELEASE + name, dst, size)
        _p(f"  [完成] {name}  {dst.stat().st_size/1024/1024:.2f} MB  "
           f"用时 {time.time()-t0:.0f}s")


# ---------------- 样例图 ----------------

# 官方仓库 input/style/ 下的经典油画（公有领域或可自由分发）。第三方图片不入库，
# 需要时用一条命令取回本地。
SAMPLE_STYLES = ("la_muse.jpg", "scene_de_rue.jpg", "picasso_self_portrait.jpg")


def cmd_fetch_samples(args):
    """从官方仓库取几张风格图（走 GitHub contents API，base64 内联返回）。"""
    import base64
    import json as _json
    import urllib.request

    SAMPLES.mkdir(parents=True, exist_ok=True)
    api = ("https://api.github.com/repos/naoto0804/pytorch-AdaIN/contents/"
           "input/style/{}")
    for name in SAMPLE_STYLES:
        dst = SAMPLES / name
        if dst.exists():
            _p(f"  [跳过] {name}")
            continue
        try:
            req = urllib.request.Request(api.format(name),
                                         headers={"User-Agent": "ShiftLite-AdaIN/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                j = _json.loads(r.read())
            dst.write_bytes(base64.b64decode(j["content"]))
            _p(f"  [取得] {name}（{dst.stat().st_size/1024:.0f} KB）")
        except Exception as e:
            _p(f"  [失败] {name}：{type(e).__name__}: {str(e)[:100]}")

    default_content = IMPL / "artifacts" / "samples" / "horse_01.png"
    _p(f"  内容图默认取主线样例：{default_content}"
       f"（{'存在' if default_content.exists() else '不存在，需先跑主线的 setup'}）")
    _p("  也可以自行指定：python stylize/run.py poc --content 图A --style 图B")


# ---------------- 出图 ----------------

def _pair_defaults() -> tuple[Path, Path]:
    """默认用主线的马图 + 官方 la_muse 油画；缺哪张就报错说明。"""
    content = IMPL / "artifacts" / "samples" / "horse_01.png"
    style = SAMPLES / "la_muse.jpg"
    return content, style


def cmd_poc(args):
    from PIL import Image
    from stylize.adain import load_stylizer, resize_style_to, to_image, to_tensor

    content_p = Path(args.content) if args.content else _pair_defaults()[0]
    style_p = Path(args.style) if args.style else _pair_defaults()[1]
    for p in (content_p, style_p):
        if not p.exists():
            raise SystemExit(f"找不到图片：{p}")

    net = load_stylizer()
    content_img = Image.open(content_p).convert("RGB")
    style_img = Image.open(style_p).convert("RGB")
    x = to_tensor(content_img, args.size)
    s = resize_style_to(x, to_tensor(style_img))

    OUT.mkdir(parents=True, exist_ok=True)
    with torch_no_grad():
        t0 = time.perf_counter()
        y = net(x, s, args.alpha)
        ms = (time.perf_counter() - t0) * 1000

    _p(f"输入 {content_p.name} ×{tuple(x.shape[-2:])}  风格 {style_p.name}  "
       f"alpha={args.alpha}")
    _p(f"PyTorch FP32 单次前向：{ms:.1f} ms")

    stem = f"{content_p.stem}__{style_p.stem}"
    out_full = OUT / f"{stem}.png"
    to_image(y).save(out_full)
    _p(f"输出：{out_full}")

    # 四联对照：内容 / 风格 / 输出 / alpha=0（重建，用于自检 decoder 没坏）
    if args.grid:
        out_img = to_image(y)
        w, h = out_img.size
        canvas = Image.new("RGB", (w * 4, h), "white")
        with torch_no_grad():
            recon = to_image(net(x, s, 0.0))
        cells = [content_img.resize((w, h)), style_img.resize((w, h)), out_img, recon]
        for i, im in enumerate(cells):
            canvas.paste(im, (i * w, 0))
        grid = OUT / f"{stem}_grid.png"
        canvas.save(grid)
        _p(f"四联对照：{grid}（左起：内容 / 风格 / alpha=1 / alpha=0 重建）")


class torch_no_grad:
    """避免在顶层 import torch 时就要装好一切；同时保持 no_grad 语义。"""

    def __enter__(self):
        import torch
        self._cm = torch.no_grad()
        self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


# ---------------- 转换 ----------------

def _build_ir(out_xml: Path, size: int = IR_SIZE):
    import numpy as np
    import openvino as ov
    import torch
    from stylize.adain import load_stylizer

    net = load_stylizer()
    x = torch.zeros(1, 3, size, size)
    a = torch.tensor([1.0])
    with torch.no_grad():
        traced = torch.jit.trace(net, (x, x, a), strict=False)
    model = ov.convert_model(traced, example_input=(x, x, a))
    model.reshape({0: [1, 3, size, size], 1: [1, 3, size, size], 2: [1]})
    # 给输入起可读名字，NNCF 校准时按名字喂数据
    model.input(0).set_names({"content"})
    model.input(1).set_names({"style"})
    model.input(2).set_names({"alpha"})
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    # compress_to_fp16 默认是 True —— 不关掉的话，这份「FP32 基线 IR」其实存的是
    # fp16 权重，锁 INFERENCE_PRECISION_HINT=f32 也救不回精度（实测 max|Δ| 0.0146），
    # 后面的「量化引入多少误差」就被这层压缩污染了。
    ov.save_model(model, str(out_xml), compress_to_fp16=False)
    return model, net


def cmd_convert(args):
    import numpy as np
    import openvino as ov
    from PIL import Image
    from stylize.adain import load_stylizer, resize_style_to, to_tensor

    xml = IR / f"adain_{args.size}.xml"
    model, _ = _build_ir(xml, args.size)
    _p(f"IR 已写入：{xml}（{xml.with_suffix('.bin').stat().st_size/1024/1024:.1f} MB）")

    net = load_stylizer()
    content_p, style_p = _pair_defaults()
    x = to_tensor(Image.open(content_p).convert("RGB"), args.size)
    s = resize_style_to(x, to_tensor(Image.open(style_p).convert("RGB")))

    import torch
    with torch.no_grad():
        y_t = net(x, s).numpy()

    core = ov.Core()
    report = {"ir": str(xml), "size": args.size, "node_count": len(model.get_ordered_ops())}
    for label, cfg in (("default", {}), ("f32", {"INFERENCE_PRECISION_HINT": "f32"})):
        m = core.compile_model(str(xml), "CPU", cfg)
        y_o = np.asarray(m({"content": x.numpy(), "style": s.numpy(), "alpha": A1})[0])
        d = np.abs(y_t - y_o)
        rng = float(y_t.max() - y_t.min()) or 1.0
        report[label] = {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "max_rel_pct": float(d.max() / rng * 100),
            "precision": str(m.get_property("INFERENCE_PRECISION_HINT")),
        }
        _p(f"  {label:8s} max|Δ|={d.max():.6f}  相对输出幅度={d.max()/rng*100:.4f}%  "
           f"实际精度={m.get_property('INFERENCE_PRECISION_HINT')}")

    (ART / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"  → {ART/'convert_report.json'}")


# ---------------- 量化 ----------------

def _calib_items(n: int, size: int = IR_SIZE):
    """校准集：主线的马图（内容）+ 本目录几张油画（风格），成对喂给双输入模型。"""
    import numpy as np
    from PIL import Image
    from stylize.adain import resize_style_to, to_tensor

    contents = sorted((IMPL / "artifacts" / "samples").glob("horse_*.png"))
    styles = sorted(SAMPLES.glob("*.jpg"))
    if not contents or not styles:
        raise SystemExit("校准集素材不足：需要 artifacts/samples/horse_*.png 与 stylize/samples/*.jpg")

    items = []
    for i in range(n):
        c = to_tensor(Image.open(contents[i % len(contents)]).convert("RGB"), size)
        s = resize_style_to(c, to_tensor(Image.open(styles[i % len(styles)]).convert("RGB")))
        items.append({"content": c.numpy(), "style": s.numpy(), "alpha": A1})
    return items


def cmd_quantize(args):
    import nncf
    import numpy as np
    import openvino as ov
    import torch
    from PIL import Image
    from stylize.adain import load_stylizer, to_image

    xml = IR / f"adain_{args.size}.xml"
    if not xml.exists():
        _p("还没有 IR，先跑 convert …")
        cmd_convert(args)
    model = ov.Core().read_model(str(xml))

    items = _calib_items(args.calib, args.size)
    _p(f"校准集：{len(items)} 对（内容=主线马图，风格=本地油画）")

    core = ov.Core()
    base = core.compile_model(str(xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    base_out = [np.asarray(base({"content": it["content"], "style": it["style"], "alpha": it["alpha"]})[0])
                for it in items[: args.eval_n]]

    def _score(outs):
        from core.metrics import psnr, ssim
        ps, ss = [], []
        for y, r in zip(outs, base_out):
            a = to_image(torch.from_numpy(y))
            b = to_image(torch.from_numpy(r))
            ps.append(psnr(a, b))
            ss.append(ssim(np.asarray(a, dtype="float32"),
                           np.asarray(b, dtype="float32")))
        return float(np.mean(ps)), float(np.mean(ss))

    _p("对照基准 = OpenVINO FP32 输出（不是 PyTorch；PyTorch→OV 的差异已在 convert 里单独验过）")
    results = {"size": args.size, "calib_pairs": len(items),
               "eval_pairs": args.eval_n, "variants": {}}

    calib = nncf.Dataset(items, lambda it: it)
    variants: list[tuple[str, object]] = [("default", None)]
    if args.exclude_norm:
        # AdaIN 的归一化在 IR 里被拆成原语，没有单个 MVN 可排除，
        # 所以这里做三组排除：按算子类型（轻）、按算子类型（全）、按节点名（正则）。
        variants += [
            ("skip_stats", nncf.IgnoredScope(types=["ReduceMean", "Sqrt"])),
            ("skip_norm_ops",
             nncf.IgnoredScope(types=["ReduceMean", "Sqrt", "Subtract", "Divide"])),
            ("skip_norm_names",
             nncf.IgnoredScope(patterns=[".*aten::std.*", ".*aten::mean.*"])),
        ]

    _p(f"{'变体':<18}{'PSNR(dB)':>10}{'SSIM':>8}{'体积(MB)':>10}")
    for tag, scope in variants:
        try:
            kw = {"subset_size": len(items),
                  "preset": nncf.QuantizationPreset.MIXED}
            if scope is not None:
                kw["ignored_scope"] = scope
            q = nncf.quantize(model, calib, **kw)
            qxml = IR / f"adain_{args.size}_int8_{tag}.xml"
            ov.save_model(q, str(qxml), compress_to_fp16=False)
            m = core.compile_model(str(qxml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            outs = [np.asarray(m({"content": it["content"], "style": it["style"], "alpha": it["alpha"]})[0])
                    for it in items[: args.eval_n]]
            psnr_v, ssim_v = _score(outs)
            size = (qxml.stat().st_size + qxml.with_suffix(".bin").stat().st_size) / 1024 / 1024
            _p(f"{tag:<18}{psnr_v:>10.2f}{ssim_v:>8.4f}{size:>10.2f}")
            # 存一张对照图：数值之外也让人能一眼看出掉点是什么样子
            OUT.mkdir(parents=True, exist_ok=True)
            to_image(torch.from_numpy(outs[0])).save(OUT / f"int8_{tag}.png")
            results["variants"][tag] = {
                "psnr_vs_ov_f32": round(psnr_v, 3),
                "ssim_vs_ov_f32": round(ssim_v, 4),
                "size_mb": round(size, 2),
                "xml": qxml.name,
            }
        except Exception as e:
            _p(f"{tag:<18}{'失败':>10}  {type(e).__name__}: {str(e)[:120]}")
            results["variants"][tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"quantize_report_{args.size}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"  → {ART/f'quantize_report_{args.size}.json'}")


# ---------------- 延迟 ----------------

def cmd_bench(args):
    import numpy as np
    import openvino as ov
    import torch
    from PIL import Image
    from stylize.adain import load_stylizer, resize_style_to, to_tensor

    content_p, style_p = _pair_defaults()
    x = to_tensor(Image.open(content_p).convert("RGB"), args.size)
    s = resize_style_to(x, to_tensor(Image.open(style_p).convert("RGB")))

    rows = []
    net = load_stylizer()
    with torch.no_grad():
        for _ in range(2):
            net(x, s)
        ts = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            net(x, s)
            ts.append((time.perf_counter() - t0) * 1000)
    rows.append(("A", "PyTorch FP32", ts))

    core = ov.Core()
    candidates = [("B", "OpenVINO FP32", IR / f"adain_{args.size}.xml",
                   {"INFERENCE_PRECISION_HINT": "f32"})]
    # INT8 有多个变体时逐个测：默认量化与「排除归一化」的加速比可能不同，
    # 只报默认那一个会掩盖推荐配置的真实代价。
    label = {"default": "INT8（全量化）", "skip_stats": "INT8（排除统计量）",
             "skip_norm_ops": "INT8（排除归一化算子）",
             "skip_norm_names": "INT8（排除归一化节点）"}
    for p in sorted(IR.glob(f"adain_{args.size}_int8_*.xml")):
        tag = p.stem.split("_int8_", 1)[1]
        candidates.append(("C", f"OpenVINO {label.get(tag, 'INT8 ' + tag)}", p,
                           {"INFERENCE_PRECISION_HINT": "f32"}))

    for tag, name, path, cfg in candidates:
        if not path.exists():
            _p(f"  [跳过] {name}：{path.name} 不存在")
            continue
        m = core.compile_model(str(path), "CPU", cfg)
        for _ in range(2):
            m({"content": x.numpy(), "style": s.numpy(), "alpha": A1})
        ts = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            m({"content": x.numpy(), "style": s.numpy(), "alpha": A1})
            ts.append((time.perf_counter() - t0) * 1000)
        size = path.stat().st_size / 1024 / 1024
        binp = path.with_suffix(".bin")
        if binp.exists():
            size += binp.stat().st_size / 1024 / 1024
        rows.append((tag, f"{name}（{size:.1f} MB）", ts))

    base = np.median(rows[0][2]) if rows else 1.0
    _p(f"{'组':<3}{'后端':<34}{'中位(ms)':>10}{'均值(ms)':>10}{'FPS':>8}{'相对A':>8}")
    for tag, name, ts in rows:
        med, mean = float(np.median(ts)), float(np.mean(ts))
        _p(f"{tag:<3}{name:<34}{med:>10.1f}{mean:>10.1f}{1000/med:>8.1f}{base/med:>7.2f}x")

    (ART / f"bench_{args.size}.json").write_text(json.dumps(
        [{"group": t, "backend": n, "samples_ms": [round(v, 2) for v in ts],
          "median_ms": round(float(np.median(ts)), 2)} for t, n, ts in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"  → {ART/f'bench_{args.size}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AdaIN 支线（issue #32）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="下载官方权重（约 94 MB）")
    d.set_defaults(func=cmd_download)

    fs = sub.add_parser("fetch-samples", help="取几张官方风格图到 stylize/samples/")
    fs.set_defaults(func=cmd_fetch_samples)

    p = sub.add_parser("poc", help="内容图 + 风格图 → 出图")
    p.add_argument("--content", default=None)
    p.add_argument("--style", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--size", type=int, default=None,
                   help="把短边缩放到该值；默认保持原分辨率（会向下对齐到 8 的倍数）")
    p.add_argument("--grid", action="store_true", help="额外输出四联对照图")
    p.set_defaults(func=cmd_poc)

    c = sub.add_parser("convert", help="转 OpenVINO IR 并验证一致性")
    c.add_argument("--size", type=int, default=IR_SIZE)
    c.set_defaults(func=cmd_convert)

    q = sub.add_parser("quantize", help="NNCF PTQ（可对照「排除归一化」）")
    q.add_argument("--size", type=int, default=IR_SIZE)
    q.add_argument("--calib", type=int, default=16)
    q.add_argument("--eval-n", type=int, default=4)
    q.add_argument("--exclude-norm", action="store_true")
    q.set_defaults(func=cmd_quantize)

    b = sub.add_parser("bench", help="A/B/C 三组延迟")
    b.add_argument("--size", type=int, default=IR_SIZE)
    b.add_argument("--rounds", type=int, default=8)
    b.set_defaults(func=cmd_bench)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
