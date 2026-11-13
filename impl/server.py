"""ShiftLite 本地可视化服务（仅标准库）。

启动：python cli.py serve
默认只绑定 127.0.0.1:8770。
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ART = ROOT / "artifacts"
UPLOADS = ART / "uploads"

# AdaIN 支线（issue #32）：与主线口径隔离，产物只落 artifacts/adain/
STYLIZE_DIR = ROOT / "stylize"
STYLIZE_WEIGHTS = STYLIZE_DIR / "weights"
STYLIZE_SAMPLES = STYLIZE_DIR / "samples"
STY_ART = ART / "adain"
STY_IR = STY_ART / "ir"
STY_OUT = STY_ART / "out"

ALLOWED_PREFIX = ("web", "artifacts", "stylize")

_runtime = None
_runtime_pair = None
_runtime_lock = threading.Lock()
_infer_lock = threading.Lock()

# 同时驻留的 Runtime 上限。实测每组 RSS 增量约 0.3–0.5 GB，
# 四个 pair 全驻留会吃掉 1.5 GB 以上，所以只保留当前这一组，
# 切 pair 时按需重编译（约 2–4 秒，仅切换时发生一次）。
_MAX_RESIDENT = 1

_sty_lock = threading.Lock()
_sty_cache: dict = {}      # {"net": torch 模型, "ov": {标签: compiled}}


def _pair_registry():
    from core.pairs import list_pairs, load_pair
    out = {}
    for pid in list_pairs():
        try:
            out[pid] = load_pair(pid)
        except Exception:
            continue
    if not out:
        from core.pairs import DEFAULT_PAIR
        out = {DEFAULT_PAIR: load_pair(DEFAULT_PAIR)}
    return out


PAIRS = _pair_registry()


def get_pair(pair_id: str | None = None):
    from core.pairs import DEFAULT_PAIR
    pid = pair_id or DEFAULT_PAIR
    if pid not in PAIRS:
        raise KeyError(f"未知迁移对：{pid}")
    return PAIRS[pid]


def get_runtime(pair_id: str | None = None):
    global _runtime, _runtime_pair
    pair = get_pair(pair_id)
    with _runtime_lock:
        if _runtime is not None and _runtime_pair == pair.id:
            return _runtime
        from core.backends import Runtime
        _runtime = Runtime(pair.weights, pair.xml, pair.int8_xml)
        _runtime_pair = pair.id
        return _runtime


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_samples(pair) -> list[dict]:
    d = pair.paths.samples
    if not d.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in exts and not p.name.startswith("_"):
            out.append({
                "id": p.stem,
                "name": p.stem.replace("_", " "),
                "url": f"/{p.relative_to(ROOT).as_posix()}",
                "kind": ("input" if p.stem.startswith(pair.input_prefix)
                         else "ref" if p.stem.startswith(pair.ref_prefix)
                         else "other"),
            })
    return out


def build_state(pair_id: str | None = None) -> dict:
    pair = get_pair(pair_id)
    rt = get_runtime(pair.id)
    avail = rt.availability()
    results = pair.paths.results
    models = pair.paths.models
    bench = _read_json(results / "benchmark.json")
    quality = _read_json(results / "quality.json")
    conv = _read_json(models / "convert_report.json")
    quant = _read_json(models / "quantize_report.json")

    groups_map = {g["id"]: g for g in (bench or {}).get("groups", [])}
    from core.backends import GROUP_LABEL
    groups = []
    for gid in ["A", "B", "C"]:
        g = dict(groups_map.get(gid, {}))
        g.setdefault("id", gid)
        g.setdefault("name", GROUP_LABEL[gid])
        g["available"] = bool(avail.get(gid))
        g["size_mb"] = rt.model_size_mb(gid)
        # C 组是量化模型，INFERENCE_PRECISION_HINT 反映的是未量化部分的提示，
        # 直接显示成 float32 会误导，这里按量化报告改写成真实精度形态。
        if gid == "C" and quant:
            g["precision"] = "int8（未量化层保持 f32）"
        groups.append(g)

    layers = _read_json(results / "layers.json")
    experiments = _experiments(conv, quant, bench, quality, avail, layers=layers)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": (bench or {}).get("device", "-"),
        "platform": (bench or {}).get("platform", "-"),
        "protocol": (bench or {}).get("protocol"),
        "pair": {
            "id": pair.id,
            "label": pair.label,
            "type": pair.type,
            "type_label": pair.type_label,
            "input_prefix": pair.input_prefix,
            "ref_prefix": pair.ref_prefix,
            "input_dir": pair.input_dir,
            "ref_dir": pair.ref_dir,
            "metrics": pair.metrics,
        },
        "pairs": [
            {"id": p.id, "label": p.label, "type": p.type,
             "type_label": p.type_label,
             "ready": (p.xml.exists() and p.int8_xml.exists())}
            for p in PAIRS.values()
        ],
        "groups": groups,
        "samples": list_samples(pair),
        "convert_report": conv,
        "quantize_report": quant,
        "quality": quality,
        "layers": layers,
        "experiments": experiments,
    }


def _experiments(conv, quant, bench, quality, avail, layers=None) -> list[dict]:
    """实验编号与 docs/PLAN.md §6 保持一致（E0–E5）。

    apple2orange 扩展已按收敛决定取消，不再占用编号；任务型斑马判别与结构/背景
    保真属于 E3 的组成部分（PLAN §4），挂在 E3 卡片下作子项，不单列编号。
    """

    def st(ok, partial=False):
        return "done" if ok else ("partial" if partial else "todo")

    deep_layer = layers if layers is not None else _read_json(ART / "results" / "layers.json")
    quality = quality or {}
    q_deep = quality.get("deep") or {}
    q_real = quality.get("metrics_source") == "real"

    def layers_note():
        if not deep_layer:
            return "未开始：需跑 python cli.py layers"
        # 逐层漂移可能落在噪声水平内，如实报道这一情形
        summary = deep_layer.get("summary", {})
        worst = summary.get("most_drifted_layer")
        cos = summary.get("most_drifted_cosine")
        monotone = summary.get("drift_monotonic_along_depth")
        trend = "漂移随深度单调累积" if monotone else "漂移未呈现单调趋势"
        if worst and isinstance(cos, (int, float)) and cos > 0.9999:
            return (f"已完成：漂移最大的 {worst} 余弦 {cos:.6f}，"
                    f"差异落在数值噪声水平内，属负结果，按期如实报道")
        return f"已完成：漂移最大层 {worst}（余弦 {cos}），{trend}"

    def quality_note():
        if not quality:
            return "待测"
        base = "SSIM / PSNR / MAE 已算；"
        if q_real:
            parts = []
            if q_deep.get("lpips"):
                parts.append("真实 LPIPS")
            if q_deep.get("fid"):
                parts.append("FID/KID")
            if q_deep.get("mask"):
                parts.append("掩码 IoU + 背景 SSIM")
            if q_deep.get("zebra_discrimination"):
                parts.append("斑马判别")
            if q_deep.get("anomalies"):
                parts.append(f"异常登记 {q_deep['anomalies']['count']} 例")
            return base + "已接真实感知指标：" + "、".join(parts)
        return (base + "真实 LPIPS 与 FID/KID、掩码 IoU、斑马判别、异常登记"
                "待跑 quality --deep（当前为代理值，不得写进报告）")

    def conv_note():
        if not conv:
            return "等待转换"
        gray = conv.get("max_gray")
        img = conv.get("input_image")
        if gray is not None:
            ok = "判据通过" if conv.get("pass") else "判据未通过"
            return (f"max|Δ|={gray} 灰度级（{ok}）；"
                    f"输入 {img or '固定随机输入'}；"
                    f"{conv.get('total_ops', '?')} 节点直转")
        return f"max|Δ|={conv.get('max_abs')}（旧口径，建议用 setup 重生成）"

    return [
        {"code": "E0", "title": "PoC：转换链路验证",
         "half": "阶段二",
         "status": st(bool(conv)),
         "note": conv_note()},
        {"code": "E1", "title": "基线出图 + 可运行 Demo",
         "half": "阶段二",
         "status": st(avail.get("A", False)),
         "note": "A 组可用，样例图与 Demo 已通" if avail.get("A") else "等待权重"},
        {"code": "E2", "title": "A/B/C 三组性能对比",
         "half": "阶段二",
         "status": st(bool(bench)),
         "note": ("已完成 " + str((bench or {}).get("generated_at", ""))) if bench else "待测"},
        {"code": "E3", "title": "量化质量代价（E3）",
         "half": "阶段二 / 三",
         "status": st(q_real, bool(quality)),
         "note": quality_note()},
        {"code": "E4", "title": "层间分工分析（逐层表示漂移）",
         "half": "阶段二",
         "status": st(bool(deep_layer)),
         "note": layers_note()},
        {"code": "E5", "title": "训练复现（独立于主对照）",
         "half": "阶段三",
         "status": "todo",
         "note": "自训权重不进 A/B/C 三组，单独成表"},
    ]


# ---------- AdaIN 支线（issue #32） ----------

def _find_img(base: Path, stem: str) -> Path | None:
    """按 stem 在目录里找图（扩展名不定）。"""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _sty_weights_ready() -> tuple[bool, str]:
    need = [STYLIZE_WEIGHTS / "vgg_normalised.pth", STYLIZE_WEIGHTS / "decoder.pth"]
    miss = [p.name for p in need if not p.exists()]
    if miss:
        return False, ("缺少 AdaIN 权重：" + "、".join(miss)
                       + "（运行 python cli.py stylize download）")
    return True, ""


def stylize_state() -> dict:
    ok, msg = _sty_weights_ready()
    styles = []
    if STYLIZE_SAMPLES.exists():
        for p in sorted(STYLIZE_SAMPLES.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                styles.append({
                    "id": p.stem,
                    "name": p.stem.replace("_", " "),
                    "url": f"/stylize/samples/{p.name}",
                })
    contents = []
    # 内容图取默认迁移对的 samples 目录（#24 之后路径是 pair 派生的，没有全局 SAMPLES 了）。
    # URL 用相对 ROOT 的真实路径，非默认 pair 也能正确指向。
    samples_dir = get_pair(None).paths.samples
    if samples_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        for p in sorted(samples_dir.iterdir()):
            if p.suffix.lower() in exts and not p.name.startswith("_"):
                contents.append({
                    "id": p.stem,
                    "source": "samples",
                    "url": f"/{p.relative_to(ROOT).as_posix()}",
                })

    backends = [{"id": "torch", "name": "PyTorch FP32"}]
    if (STY_IR / "adain_256.xml").exists():
        backends.append({"id": "ov", "name": "OpenVINO FP32"})
    if (STY_IR / "adain_256_int8_skip_norm_ops.xml").exists():
        backends.append({"id": "int8", "name": "OpenVINO INT8（排除归一化）"})

    return {
        "ready": ok,
        "message": msg,
        "styles": styles,
        "contents": contents,
        "backends": backends if ok else [],
        "note": ("AdaIN 支线：换的是画风，不改内容语义；与主线 A/B/C 三组指标不并列比较。"
                 "OV / INT8 后端固定 256 输入；PyTorch 后端保持原分辨率。"),
    }


def _sty_net():
    """PyTorch 模型缓存（约 90 MB 权重，首次加载 ~1.5 s）。"""
    if "net" not in _sty_cache:
        from stylize.adain import load_stylizer
        _sty_cache["net"] = load_stylizer()
    return _sty_cache["net"]


def _sty_ov(xml: Path):
    key = f"ov:{xml.name}"
    if key not in _sty_cache:
        import openvino as ov
        _sty_cache[key] = ov.Core().compile_model(
            str(xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    return _sty_cache[key]


def run_stylize(content_path: Path, style_path: Path, alpha: float,
                backend: str, out_path: Path) -> float:
    """跑一次风格迁移，返回耗时（ms）。backend ∈ torch / ov / int8。"""
    import numpy as np
    import torch
    from PIL import Image
    from stylize.adain import resize_style_to, to_image, to_tensor

    content = Image.open(content_path).convert("RGB")
    style = Image.open(style_path).convert("RGB")

    t0 = time.perf_counter()
    if backend == "torch":
        net = _sty_net()
        x = to_tensor(content)
        s = resize_style_to(x, to_tensor(style))
        with torch.no_grad():
            out_img = to_image(net(x, s, float(alpha)))
    else:
        xml = STY_IR / ("adain_256.xml" if backend == "ov"
                        else "adain_256_int8_skip_norm_ops.xml")
        if not xml.exists():
            raise FileNotFoundError(f"缺少 IR：{xml.name}（先跑 cli.py stylize convert）")
        m = _sty_ov(xml)
        x = to_tensor(content, 256)
        s = resize_style_to(x, to_tensor(style, 256))
        out = np.asarray(m({
            "content": x.numpy(),
            "style": s.numpy(),
            "alpha": np.array([float(alpha)], dtype="float32"),
        })[0])
        out_img = to_image(torch.from_numpy(out))

    ms = (time.perf_counter() - t0) * 1000
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    return ms


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ShiftLite/1.0"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    # ---------- 基础响应 ----------

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, code: int, msg: str):
        self._json({"ok": False, "error": msg}, code)

    def _safe_path(self, rel: str) -> Path | None:
        rel = urllib.parse.unquote(rel).lstrip("/")
        norm = (ROOT / rel).resolve()
        try:
            norm.relative_to(ROOT.resolve())
        except ValueError:
            return None
        if not norm.is_file():
            return None
        return norm

    # ---------- 路由 ----------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._serve_file(WEB / "index.html")

        if path == "/api/state":
            q = urllib.parse.parse_qs(parsed.query)
            pid = (q.get("pair") or [None])[0]
            try:
                return self._json({"ok": True, "state": build_state(pid)})
            except KeyError as e:
                return self._err(404, str(e).strip("'\""))
            except Exception as e:
                return self._err(500, f"状态构建失败：{e}")

        if path == "/api/stylize/state":
            try:
                return self._json({"ok": True, "state": stylize_state()})
            except Exception as e:
                return self._err(500, f"支线状态构建失败：{e}")

        if path.startswith("/artifacts/") or path.startswith("/web/") \
                or path.startswith("/stylize/"):
            fp = self._safe_path(path)
            if not fp:
                return self._err(404, "文件不存在")
            return self._serve_file(fp)

        return self._err(404, "路径不存在")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/infer":
            return self._api_infer()
        if parsed.path == "/api/upload":
            return self._api_upload()
        if parsed.path == "/api/stylize":
            return self._api_stylize()
        return self._err(404, "路径不存在")

    # ---------- 文件 ----------

    def _serve_file(self, fp: Path):
        if not fp.exists():
            return self._err(404, "文件不存在")
        ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
        if fp.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        if fp.suffix in (".html", ".css"):
            ctype += "; charset=utf-8"
        data = fp.read_bytes()
        self._send(200, data, ctype, {"Cache-Control": "no-cache"})

    # ---------- 推理 ----------

    def _api_infer(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 32 * 1024 * 1024:
                return self._err(413, "请求体过大")
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._err(400, f"请求解析失败：{e}")

        sample = str(req.get("sample", "")).strip()
        group = str(req.get("group", "A")).upper()
        source = req.get("source") or "samples"
        pair_id = req.get("pair") or None

        if group not in ("A", "B", "C"):
            return self._err(400, "group 只能是 A / B / C")
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", sample):
            return self._err(400, "sample 参数非法")

        try:
            pair = get_pair(pair_id)
        except KeyError as e:
            return self._err(400, str(e).strip("'\""))

        base = UPLOADS if source == "uploads" else pair.paths.samples
        img_path = base / f"{sample}.png"
        if not img_path.exists():
            for ext in (".jpg", ".jpeg", ".webp"):
                if (base / f"{sample}{ext}").exists():
                    img_path = base / f"{sample}{ext}"
                    break
            else:
                return self._err(404, "找不到输入图")

        rt = get_runtime(pair.id)
        if not rt.availability().get(group):
            return self._err(409, f"{group} 组模型尚未生成（{pair.label}）")

        from core.utils import load_image, to_tensor, to_image

        try:
            with _infer_lock:
                img = load_image(img_path)
                x = to_tensor(img)
                # 预热 2 轮：编译后首几次推理仍有惰性初始化，只预热 1 轮会让
                # 单次显示的延迟明显高于 bench 的稳态值
                rt.warmup(group, 2)
                t0 = time.perf_counter()
                out = rt.infer(x, group)
                ms = (time.perf_counter() - t0) * 1000
                result = to_image(out)

            infer_out = pair.paths.results / "inference"
            infer_out.mkdir(parents=True, exist_ok=True)
            dst = infer_out / f"{sample}_{group}.png"
            result.save(dst)
            ts = int(time.time() * 1000)

            return self._json({
                "ok": True,
                "pair": pair.id,
                "group": group,
                "latency_ms": round(ms, 2),
                "fps": round(1000 / ms, 2) if ms else 0,
                "input_url": f"/{img_path.relative_to(ROOT).as_posix()}",
                "output_url": f"/{dst.relative_to(ROOT).as_posix()}?t={ts}",
            })
        except Exception as e:
            return self._err(500, f"推理失败：{type(e).__name__}: {e}")

    def _api_stylize(self):
        ok, msg = _sty_weights_ready()
        if not ok:
            return self._err(409, msg)

        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 32 * 1024 * 1024:
                return self._err(413, "请求体过大")
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._err(400, f"请求解析失败：{e}")

        content = req.get("content") or {}
        cid = str(content.get("id", "")).strip()
        csrc = content.get("source") or "samples"
        sid = str(req.get("style", "")).strip()
        backend = str(req.get("backend", "torch"))
        try:
            alpha = max(0.0, min(1.0, float(req.get("alpha", 1.0))))
        except (TypeError, ValueError):
            return self._err(400, "alpha 非法")

        for label, val in (("内容图", cid), ("风格图", sid)):
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", val):
                return self._err(400, f"{label}参数非法")
        if backend not in ("torch", "ov", "int8"):
            return self._err(400, "backend 只能是 torch / ov / int8")
        if csrc not in ("samples", "uploads"):
            return self._err(400, "content.source 只能是 samples / uploads")

        cbase = UPLOADS if csrc == "uploads" else get_pair(None).paths.samples
        cpath = _find_img(cbase, cid)
        if cpath is None:
            return self._err(404, "找不到内容图")
        spath = _find_img(STYLIZE_SAMPLES, sid)
        if spath is None:
            return self._err(404, "找不到风格图")

        ts = int(time.time() * 1000)
        out = STY_OUT / f"api_{cid}__{sid}_{ts}.png"
        try:
            with _sty_lock:
                ms = run_stylize(cpath, spath, alpha, backend, out)
        except Exception as e:
            return self._err(500, f"风格化失败：{type(e).__name__}: {e}")

        return self._json({
            "ok": True,
            "backend": backend,
            "alpha": round(alpha, 2),
            "latency_ms": round(ms, 2),
            "fps": round(1000 / ms, 2) if ms else 0,
            "content_url": f"/{cpath.relative_to(ROOT).as_posix()}",
            "style_url": f"/{spath.relative_to(ROOT).as_posix()}",
            "output_url": f"/{out.relative_to(ROOT).as_posix()}?t={ts}",
        })

    def _api_upload(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except Exception:
            return self._err(400, "缺少 Content-Length")
        if n <= 0 or n > 32 * 1024 * 1024:
            return self._err(413, "文件大小超出限制（≤32MB）")

        raw = self.rfile.read(n)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        name = f"up_{int(time.time()*1000)}"
        tmp = UPLOADS / f"_{name}.tmp"
        tmp.write_bytes(raw)
        try:
            from PIL import Image
            with Image.open(tmp) as im:
                im.convert("RGB").save(UPLOADS / f"{name}.png")
        except Exception:
            tmp.unlink(missing_ok=True)
            return self._err(400, "不是可识别的图片")
        tmp.unlink(missing_ok=True)

        return self._json({
            "ok": True,
            "id": name,
            "url": f"/artifacts/uploads/{name}.png",
        })


def run(host: str = "127.0.0.1", port: int = 8770):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"ShiftLite 可视化界面已启动： http://{host}:{port}")
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()
