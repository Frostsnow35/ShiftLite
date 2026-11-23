# -*- coding: utf-8 -*-
"""生成 #20 复核样本对照拼图（X/A/B/C 四列）+ 复核 CSV 骨架。

复现 2026-11-23 复核所用拼图；已有 human_review_pending85.csv 时不会覆盖人工结论。
"""
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:/人工智能导论项目/ShiftLite")
CHECK = ROOT / "docs/experiments/2026-11-09-7940HX/official_checklist.csv"
OUT = ROOT / "docs/experiments/2026-11-23-review"
SHEETS = OUT / "contact_sheets"
QDIR = ROOT / "impl/artifacts/results/quality"
XDIR = ROOT / "impl/artifacts/eval/testA"
PANEL = 256
LABEL_H = 26
STRIPS_PER_SHEET = 5

rows = list(csv.DictReader(CHECK.open(encoding="utf-8-sig")))
pending = [r for r in rows if r["human_status"] == "pending"]
if not pending:  # 清单已清零时，按复核 CSV 的样本顺序重建拼图
    done = OUT / "human_review_pending85.csv"
    if done.exists():
        dr = list(csv.DictReader(done.open(encoding="utf-8-sig")))
        lookup = {r["sample"]: r for r in rows}
        pending = [lookup[d["sample"]] for d in dr if d["sample"] in lookup]
order = {"P1_first_failure": 0, "P2_multi_instance": 1, "P3_spotcheck": 2}
pending.sort(key=lambda r: (order.get(r["priority"], 9), r["sample"]))
print("pending:", len(pending))

font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
x_map = {}
q = json.loads((ROOT / "impl/artifacts/results/quality.json").read_text(encoding="utf-8"))
for row in q["rows"]:
    p = ROOT / "impl" / row["input"].replace("\\", "/")
    x_map[row["sample"]] = p

missing = []
SHEETS.mkdir(parents=True, exist_ok=True)
sheet_idx = 0
for start in range(0, len(pending), STRIPS_PER_SHEET):
    batch = pending[start:start + STRIPS_PER_SHEET]
    sheet = Image.new("RGB", (PANEL * 4, (PANEL + LABEL_H) * len(batch)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for j, r in enumerate(batch):
        s = r["sample"]
        paths = [x_map.get(s)] + [QDIR / f"{s}_{g}.png" for g in "ABC"]
        y0 = j * (PANEL + LABEL_H)
        draw.text((6, y0 + 4), f"{s}  [{r['priority']}]", fill=(255, 220, 80), font=font)
        for i, p in enumerate(paths):
            if p is None or not p.is_file():
                missing.append((s, str(p)))
                continue
            im = Image.open(p).convert("RGB")
            if im.size != (PANEL, PANEL):
                im = im.resize((PANEL, PANEL))
            sheet.paste(im, (i * PANEL, y0 + LABEL_H))
        for i in range(1, 4):
            draw.line([(i * PANEL, y0), (i * PANEL, y0 + LABEL_H)], fill=(90, 90, 90))
    name = f"sheet_{sheet_idx:02d}.jpg"
    sheet.save(SHEETS / name, quality=88)
    print(name, [r["sample"] for r in batch])
    sheet_idx += 1

print("missing:", missing)

# CSV 骨架：机器列保留，人工列留空待填（已有结论文件时不覆盖）
csv_path = OUT / "human_review_pending85.csv"
if csv_path.exists():
    print("csv exists, keep conclusions:", csv_path)
else:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample", "priority", "machine_reasons", "sheet", "human_status",
                    "conclusion", "notes"])
        for i, r in enumerate(pending):
            w.writerow([r["sample"], r["priority"], r["machine_reasons"],
                        f"sheet_{i // STRIPS_PER_SHEET:02d}.jpg", "pending", "", ""])
    print("csv:", csv_path)
