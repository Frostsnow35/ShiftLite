"""失真样本审计（#17）：四类归档 + 固有/量化分离 + 层漂移对照。

输入：
  --review-queue  官方 review_queue.json（逐样本检测框/失败侧/原因）
  --human-review  人工复核 CSV（可选；提供 conclusion/notes）
  --quality       质量 JSON（可选；提供逐图 ssim_bc/背景/LPIPS，补充证据）
  --layers        层间漂移 layers.json（可选；用于层组对照）
  --out-dir       输出目录（anomalies_labeled.json / .md）

口径：
  - 四类：形变 / 位置偏移 / 背景改写 / 结构异常。
  - 只统计“生成侧”候选：A/B/C 有检测失败，或有人工意见，或 ssim_bc 低于阈值。
  - X（输入马图）检测失败单独列出，不计为生成失真。
  - 多实例“主体对应未验证”是检测口径问题，默认不计为失真。
  - 来源：baseline_inherent（A/B FP32 失败）、quantization_introduced（仅 C）、
    quantization_aggravated（A/B 与 C 均失败或 ssim_bc 低）、human_reported（仅人工记录）。
  - 机器预标注，不代替人工结论。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CLASSES = ["形变", "位置偏移", "背景改写", "结构异常"]
LAYER_REF = {
    "结构异常": "down_relu_1..3（浅层结构/位置）",
    "位置偏移": "down_relu_1..3 + mask（浅层结构/位置）",
    "形变": "resblock_06..09（深层残差，风格/结构）",
    "背景改写": "up_relu_1..2 + head_conv（解码，纹理/背景）",
}
SOURCE_LABEL = {
    "baseline_inherent": "基线固有（A/B FP32 已失败）",
    "quantization_introduced": "量化引入（仅 C 失败）",
    "quantization_aggravated": "量化加重（A/B 与 C 均失败或 ssim_bc 低）",
    "human_reported": "仅人工记录",
    "other": "其他",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def load_csv(path):
    if not path or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return {r["sample"]: r for r in csv.DictReader(f)}


def box_info(box):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "area": w * h, "diag": math.hypot(w, h)}


def text_of(h):
    return f"{h.get('conclusion','')} {h.get('notes','')}" if h else ""


def classify(entry, human, qrow):
    det = entry.get("detections", {})
    failed = [s for s, d in det.items() if not d.get("found")]
    gen_failed = [s for s in failed if s in ("A", "B", "C")]
    x_failed = "X" in failed
    classes, ev = set(), []

    if gen_failed:
        reasons = sorted({d.get("reason") for s, d in det.items()
                          if s in gen_failed and not d.get("found") and d.get("reason")})
        ev.append("生成侧失败:" + ",".join(gen_failed))
        if reasons:
            ev.append("原因:" + ",".join(reasons))
        classes.add("结构异常")

    bx, bc = box_info(det.get("X", {}).get("box")), box_info(det.get("C", {}).get("box"))
    if bx and bc:
        shift = math.hypot(bx["cx"] - bc["cx"], bx["cy"] - bc["cy"]) / max(bx["diag"], 1e-6)
        area = bc["area"] / max(bx["area"], 1e-6)
        ev.append(f"X→C 中心偏移比={shift:.3f}; 面积比={area:.2f}")
        if shift > 0.15:
            classes.add("位置偏移")
        if area < 0.7 or area > 1.4:
            classes.add("形变")

    if qrow:
        if qrow.get("mask_iou_ac") is not None:
            ev.append(f"mask_iou_ac={qrow['mask_iou_ac']:.3f}")
            if qrow["mask_iou_ac"] < 0.80:
                classes.add("位置偏移")
        if qrow.get("edge_ssim_bc") is not None:
            ev.append(f"edge_ssim_bc={qrow['edge_ssim_bc']:.3f}")
            if qrow["edge_ssim_bc"] < 0.70:
                classes.add("形变")
        if qrow.get("background_ssim_ac") is not None:
            ev.append(f"background_ssim_ac={qrow['background_ssim_ac']:.3f}")
            if qrow["background_ssim_ac"] < 0.85:
                classes.add("背景改写")
        lp = qrow.get("lpips") or {}
        if isinstance(lp, dict) and lp.get("L_bc") is not None:
            ev.append(f"LPIPS_bc={lp['L_bc']:.3f}")
        if qrow.get("ssim_bc") is not None:
            ev.append(f"ssim_bc={qrow['ssim_bc']:.3f}")

    t = text_of(human)
    if any(k in t for k in ("背景", "天空", "颗粒", "纹理变化")):
        classes.add("背景改写")
    if any(k in t for k in ("形变", "扭曲", "变形", "轮廓异常", "结构异常", "畸形")):
        classes.add("形变")
    if any(k in t for k in ("位置偏移", "偏移", "移动")):
        classes.add("位置偏移")
    if any(k in t for k in ("主体对应不一致", "选错", "对应不同", "崩坏")):
        classes.add("结构异常")

    if not classes:
        classes.add("结构异常" if gen_failed else "形变")
    return sorted(classes), ev, gen_failed, x_failed


def source_of(gen_failed, quant_anom, human_seen):
    s = set()
    fp32 = any(x in ("A", "B") for x in gen_failed)
    c = "C" in gen_failed
    if fp32:
        s.add("baseline_inherent")
    if c and fp32:
        s.add("quantization_aggravated")
    elif c:
        s.add("quantization_introduced")
    if quant_anom:
        s.add("quantization_aggravated")
    if human_seen and not s:
        s.add("human_reported")
    if not s:
        s.add("other")
    return sorted(s)


def layer_summary(layers):
    s = (layers or {}).get("summary", {})
    return {
        "resblock_head_to_tail": s.get("resblock_cosine_head_tail"),
        "most_drifted": (s.get("most_drifted_layer"), s.get("most_drifted_cosine")),
        "least_drifted": (s.get("least_drifted_layer"), s.get("least_drifted_cosine")),
        "monotonic": s.get("drift_monotonic_along_depth"),
    }


def main():
    ap = argparse.ArgumentParser(description="失真样本审计（#17）")
    ap.add_argument("--review-queue", required=True)
    ap.add_argument("--human-review")
    ap.add_argument("--quality")
    ap.add_argument("--layers")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rq = load_json(args.review_queue)
    human = load_csv(args.human_review)
    quality = load_json(args.quality)
    qrows = {r["sample"]: r for r in (quality or {}).get("rows", [])}
    layers = load_json(args.layers)
    thr = ((quality or {}).get("deep", {}).get("anomalies", {}) or {}).get("threshold", 0.80)

    records, input_detect, multi_only = [], [], []
    seen = set()
    for entry in rq.get("review_queue", []):
        sid = entry["sample"]
        h = human.get(sid)
        qrow = qrows.get(sid)
        human_seen = bool(h and h.get("human_status") in ("reviewed", "needs_discussion"))
        quant_anom = bool(qrow and qrow.get("ssim_bc") is not None and qrow["ssim_bc"] < thr)
        multi = any((d.get("candidate_count") or 0) > 1 for d in entry.get("detections", {}).values())
        classes, ev, gen_failed, x_failed = classify(entry, h, qrow)
        candidate = bool(gen_failed) or human_seen or quant_anom
        if not candidate:
            if x_failed:
                input_detect.append({"sample": sid,
                                     "reasons": entry.get("reasons", []),
                                     "comparison_file": entry.get("comparison_file")})
            elif multi:
                multi_only.append(sid)
            continue
        src = source_of(gen_failed, quant_anom, human_seen)
        records.append({
            "sample": sid,
            "classes": classes,
            "source": src,
            "human_status": (h or {}).get("human_status") or entry.get("human_status") or "pending",
            "human_conclusion": (h or {}).get("conclusion"),
            "human_notes": (h or {}).get("notes", ""),
            "failed_sides": [s for s in entry.get("detections", {}) if not entry["detections"][s].get("found")],
            "gen_failed_sides": gen_failed,
            "x_failed": x_failed,
            "reasons": entry.get("reasons", []),
            "evidence": "; ".join(ev),
            "layer_ref": sorted({LAYER_REF[c] for c in classes}),
            "comparison_file": entry.get("comparison_file"),
        })
        seen.add(sid)

    # 主表 ssim_bc 异常但不在 review_queue 的样本
    for sid, row in qrows.items():
        if sid in seen or row.get("ssim_bc") is None or row["ssim_bc"] >= thr:
            continue
        classes, ev, _, _ = classify({"detections": {}}, human.get(sid), row)
        records.append({
            "sample": sid, "classes": classes, "source": ["quantization_aggravated"],
            "human_status": (human.get(sid) or {}).get("human_status", "pending"),
            "human_conclusion": (human.get(sid) or {}).get("conclusion"),
            "human_notes": (human.get(sid) or {}).get("notes", ""),
            "failed_sides": [], "gen_failed_sides": [], "x_failed": False, "reasons": [],
            "evidence": "; ".join(ev),
            "layer_ref": sorted({LAYER_REF[c] for c in classes}), "comparison_file": None,
        })

    by_source, by_class = {}, {c: 0 for c in CLASSES}
    for r in records:
        for s in r["source"]:
            by_source[s] = by_source.get(s, 0) + 1
        for c in r["classes"]:
            by_class[c] = by_class.get(c, 0) + 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "anomaly-audit-v1", "classes": CLASSES,
        "note": "机器预标注，不代替人工结论；X 侧检测失败与多实例候选不计为生成失真。",
        "inputs": {"review_queue": str(args.review_queue), "human_review": str(args.human_review),
                   "quality": str(args.quality), "layers": str(args.layers)},
        "counts": {"records": len(records), "by_source": by_source, "by_class": by_class,
                   "input_detection_failures": len(input_detect), "multi_instance_only": len(multi_only)},
        "layers": layer_summary(layers),
    }
    (out / "anomalies_labeled.json").write_text(
        json.dumps({"meta": meta, "records": records,
                    "input_detection_failures": input_detect,
                    "multi_instance_candidates": multi_only}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    L = ["# 失真样本审计（#17）", "",
         "> 机器预标注；不代替人工结论；X 侧检测失败与多实例候选不计为生成失真。", "",
         f"- 失真记录：{len(records)}",
         f"- 来源计数：{by_source}",
         f"- 四类计数：{by_class}",
         f"- 输入侧(X)检测失败（不计失真）：{len(input_detect)}",
         f"- 多实例候选（未计入）：{len(multi_only)}", "",
         "## 一、按来源", "", "| 来源 | 样本数 |", "|---|---:|"]
    for s in ("baseline_inherent", "quantization_introduced", "quantization_aggravated", "human_reported", "other"):
        L.append(f"| {SOURCE_LABEL[s]} | {by_source.get(s, 0)} |")
    L += ["", "## 二、按四类", "", "| 类别 | 次数 | 对应层组 |", "|---|---:|---|"]
    for c in CLASSES:
        L.append(f"| {c} | {by_class.get(c, 0)} | {LAYER_REF[c]} |")

    def table(title, pred):
        rows = [r for r in records if pred(r)]
        L.extend(["", f"## {title}", "", "| 样本 | 生成侧失败 | 自动类别 | 证据 | 人工 |", "|---|---|---|---|---|"])
        for r in rows:
            L.append(f"| {r['sample']} | {','.join(r['gen_failed_sides']) or '-'} | "
                     f"{','.join(r['classes'])} | {r['evidence']} | {r['human_status']} |")

    table("三、基线固有（A/B FP32 已失败）", lambda r: "baseline_inherent" in r["source"])
    table("四、量化引入/加重", lambda r: ("quantization_introduced" in r["source"] or
                                        "quantization_aggravated" in r["source"]))
    table("五、仅人工记录", lambda r: r["source"] == ["human_reported"])

    ls = meta["layers"]
    L += ["", "## 六、失真类型 × 层漂移对照", "", "| 类别 | 可能对应层组 | 依据 |", "|---|---|---|",
          f"| 结构异常 | {LAYER_REF['结构异常']} | 浅层≈0.9996 稳定，结构指标仍较高 |",
          f"| 位置偏移 | {LAYER_REF['位置偏移']} | 浅层稳定但掩码 IoU 下降 |",
          f"| 形变 | {LAYER_REF['形变']} | 残差块首尾 {ls.get('resblock_head_to_tail')}，最大漂移 {ls.get('most_drifted')} |",
          f"| 背景改写 | {LAYER_REF['背景改写']} | 解码/输出头漂移显著，对应背景 SSIM |"]

    L += ["", "## 七、输入侧(X)检测失败（不计失真）", "", "| 样本 | 原因 | 对照图 |", "|---|---|---|"]
    for r in input_detect:
        L.append(f"| {r['sample']} | {','.join(r['reasons']) or '-'} | {r['comparison_file']} |")

    L += ["", "## 八、边界", "",
          "- 自动类别由检测失败侧、X↔C 检测框变化、阈值与关键词产生，需人工复核后定稿。",
          "- 位置偏移/形变来自检测框中心/面积变化，非光流或真值。",
          "- 主表 ssim/LPIPS 与检测批次可能来自不同机器/日期，证据列已如实保留。",
          "- 多实例候选是“主体对应未验证”，不等于生成失败。"]
    (out / "anomalies_labeled.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"OK records={len(records)} input_detect={len(input_detect)} multi={len(multi_only)} out={out}")


if __name__ == "__main__":
    main()
