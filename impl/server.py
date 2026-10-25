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
SAMPLES = ART / "samples"
RESULTS = ART / "results"
MODELS = ART / "models"
UPLOADS = ART / "uploads"
INFER_OUT = RESULTS / "inference"

ALLOWED_PREFIX = ("web", "artifacts")

_runtime = None
_runtime_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            from core.backends import Runtime
            _runtime = Runtime(MODELS / "horse2zebra.pth",
                               MODELS / "horse2zebra.xml",
                               MODELS / "horse2zebra_int8.xml")
        return _runtime


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_samples() -> list[dict]:
    if not SAMPLES.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = []
    for p in sorted(SAMPLES.iterdir()):
        if p.suffix.lower() in exts and not p.name.startswith("_"):
            out.append({
                "id": p.stem,
                "name": p.stem.replace("_", " "),
                "url": f"/artifacts/samples/{p.name}",
                "kind": "zebra" if p.stem.startswith("zebra") else "horse",
            })
    return out


def build_state() -> dict:
    rt = get_runtime()
    avail = rt.availability()
    bench = _read_json(RESULTS / "benchmark.json")
    quality = _read_json(RESULTS / "quality.json")
    conv = _read_json(MODELS / "convert_report.json")
    quant = _read_json(MODELS / "quantize_report.json")

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

    layers = _read_json(RESULTS / "layers.json")
    experiments = _experiments(conv, quant, bench, quality, avail, layers=layers)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": (bench or {}).get("device", "-"),
        "platform": (bench or {}).get("platform", "-"),
        "protocol": (bench or {}).get("protocol"),
        "groups": groups,
        "samples": list_samples(),
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

    deep_layer = layers if layers is not None else _read_json(RESULTS / "layers.json")
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
            try:
                return self._json({"ok": True, "state": build_state()})
            except Exception as e:
                return self._err(500, f"状态构建失败：{e}")

        if path.startswith("/artifacts/") or path.startswith("/web/"):
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

        if group not in ("A", "B", "C"):
            return self._err(400, "group 只能是 A / B / C")
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", sample):
            return self._err(400, "sample 参数非法")

        base = UPLOADS if source == "uploads" else SAMPLES
        img_path = base / f"{sample}.png"
        if not img_path.exists():
            for ext in (".jpg", ".jpeg", ".webp"):
                if (base / f"{sample}{ext}").exists():
                    img_path = base / f"{sample}{ext}"
                    break
            else:
                return self._err(404, "找不到输入图")

        rt = get_runtime()
        if not rt.availability().get(group):
            return self._err(409, f"{group} 组模型尚未生成")

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

            INFER_OUT.mkdir(parents=True, exist_ok=True)
            dst = INFER_OUT / f"{sample}_{group}.png"
            result.save(dst)
            ts = int(time.time() * 1000)

            return self._json({
                "ok": True,
                "group": group,
                "latency_ms": round(ms, 2),
                "fps": round(1000 / ms, 2) if ms else 0,
                "input_url": f"/{img_path.relative_to(ROOT).as_posix()}",
                "output_url": f"/{dst.relative_to(ROOT).as_posix()}?t={ts}",
            })
        except Exception as e:
            return self._err(500, f"推理失败：{type(e).__name__}: {e}")

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
