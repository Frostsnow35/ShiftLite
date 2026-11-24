"""合并 #20 人工复核：#20 final merged table（可复用）。

规则：
- 以官方首轮 `human_review.csv` 为底，保留未在本轮分派样本的首轮状态（如 P3 抽查）。
- 本轮 4 份个人表覆盖已分派样本；每位样本取主复核人，另存第二位复核。
- 复核人使用组内代号（问卷人A..问卷人D），真实姓名不写入仓库。
- 机器字段只读：不改 detection/quality JSON。

用法：
  python tools/merge_review.py \
    --base-review ../human-review/human-review/human_review.csv \
    --checklist ../human-review/human-review/official_checklist.csv \
    --personal-dir <本地复核 runs 目录> \
    --failures <本地复核 runs 目录>/detection_failures_merged.csv \
    --out-dir ../docs/experiments/2026-11-09-7940HX/human_review_final
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

PRIMARY = {"n02381460_1090": "问卷人A", "n02381460_1540": "问卷人B", "n02381460_3010": "问卷人C"}
FIRST_ROUND = "2026-11-10 first round"
SECOND_ROUND = "2026-11-23 second round"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-review", required=True)
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--personal-dir", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    checklist = read_csv(args.checklist)
    base = {r["sample"]: r for r in read_csv(args.base_review)}
    personal = {}
    for p in sorted(glob.glob(str(Path(args.personal_dir) / "human_review_*.csv"))):
        if Path(p).name == "human_review_merged.csv":
            continue
        for row in read_csv(p):
            if row.get("reviewer"):
                personal.setdefault(row["sample"], []).append(row)

    merged = []
    for c in checklist:
        s = c["sample"]
        recs = personal.get(s, [])
        recs.sort(key=lambda x: 0 if x["reviewer"] == PRIMARY.get(s) else 1)
        row = {
            "priority": c["priority"], "sample": s,
            "failure_sides": c["failure_sides"], "failure_detail": c["failure_detail"],
            "machine_reasons": c["machine_reasons"], "comparison_file": c["comparison_file"],
        }
        if recs:
            p = recs[0]
            row.update(human_status=p["human_status"], reviewer=p["reviewer"],
                       date=p["date"], conclusion=p["conclusion"], notes=p["notes"],
                       round=SECOND_ROUND)
            if len(recs) > 1:
                s2 = recs[1]
                row.update(second_reviewer=s2["reviewer"], second_status=s2["human_status"],
                           second_conclusion=s2["conclusion"], second_notes=s2["notes"])
            else:
                row.update(second_reviewer="", second_status="",
                           second_conclusion="", second_notes="")
        else:
            b = base.get(s, {})
            if b.get("human_status") in ("reviewed", "needs_discussion"):
                row.update(human_status=b["human_status"],
                           reviewer=b.get("reviewer") or "首轮（署名空）",
                           date=b.get("date") or "2026-11-10",
                           conclusion=b.get("conclusion", ""), notes=b.get("notes", ""),
                           round=FIRST_ROUND)
            else:
                row.update(human_status="pending", reviewer="", date="",
                           conclusion="", notes="", round="")
            row.update(second_reviewer="", second_status="", second_conclusion="", second_notes="")
        merged.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "human_review_merged.csv", merged)

    failures = read_csv(args.failures)
    write_csv(out / "detection_failures_merged.csv", failures)

    counts = {}
    rounds = {}
    needs = []
    for r in merged:
        counts[r["human_status"]] = counts.get(r["human_status"], 0) + 1
        rounds[r["round"]] = rounds.get(r["round"], 0) + 1
        if r["human_status"] == "needs_discussion":
            needs.append(r["sample"])
    second = [r for r in merged if r["second_reviewer"]]
    coverage = {
        "total": len(merged), "by_status": counts, "by_round": rounds,
        "second_reviews": len(second),
        "needs_discussion": needs,
        "reviewer_ids": "问卷人A..问卷人D（组内评分人代号）",
    }
    (out / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# #20 人工复核覆盖统计", "",
         f"- 总计：{len(merged)}",
         f"- 状态计数：{counts}",
         f"- 轮次计数：{rounds}",
         f"- 第二位复核：{len(second)} 例",
         f"- 待讨论：{', '.join(needs) if needs else '无'}",
         "- 复核人：`问卷人A`–`问卷人D`（组内评分人代号，真实姓名不写入仓库）",
         "- `reviewed` 只表示已查看并记录，不代表质量通过。", "",
         "## 待讨论清单", "", "| 样本 | 主结论 | 第二复核 | 第二结论 |", "|---|---|---|---|"]
    for r in merged:
        if r["human_status"] == "needs_discussion":
            L.append(f"| {r['sample']} | {r['conclusion']} | {r['second_reviewer'] or '-'} | {r['second_conclusion'] or '-'} |")
    L += ["", "## 说明", "",
          "- P3 抽查 7 张沿用 2026-11-10 首轮状态；本轮未重复查看。",
          "- 16 个检测失败样本的逐侧状态见 `detection_failures_merged.csv`。",
          "- X 侧（输入马图）检测失败与生成侧失败分开记录，不把检测失败直接写成生成畸形。"]
    (out / "coverage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"OK total={len(merged)} status={counts} rounds={rounds} second={len(second)}")


if __name__ == "__main__":
    main()
