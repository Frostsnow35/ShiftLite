"""生成 #15 主观问卷的盲评材料：FP32(b) vs INT8(c) 成对、随机左右。

用法：
  python tools/build_blind_pairs.py --quality-dir artifacts/results/quality \
      --out-dir ../docs/experiments/questionnaire_T1 --n 30 --seed 20261120

产出：
  pairs/pair_XXX_L.png, pair_XXX_R.png   中性命名，不泄露 b/c
  blind_pairs.csv                        给打分人（只有 pair_id 与左右图）
  answer_key.csv                         汇总人保留（pair_id -> 样本, 左右组别）
  subjective_template.csv                打分表（4 项 × 左右）
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

ITEMS = ["naturalness", "zebra_feature", "pose_structure", "background"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    qdir = Path(args.quality_dir)
    out = Path(args.out_dir)
    pairs_dir = out / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted({p.name[:-6] for p in qdir.glob("*_B.png")}
                     & {p.name[:-6] for p in qdir.glob("*_C.png")})
    if not samples:
        raise SystemExit(f"在 {qdir} 没有找到 *_B.png / *_C.png")
    rng = random.Random(args.seed)
    picked = samples if len(samples) <= args.n else rng.sample(samples, args.n)
    picked.sort()

    pairs_rows, key_rows, tpl_rows = [], [], []
    for i, sid in enumerate(picked, 1):
        pid = f"pair_{i:03d}"
        flip = rng.random() < 0.5
        left_group, right_group = ("C", "B") if flip else ("B", "C")
        lsrc = qdir / f"{sid}_{left_group}.png"
        rsrc = qdir / f"{sid}_{right_group}.png"
        lname, rname = f"{pid}_L.png", f"{pid}_R.png"
        shutil.copyfile(lsrc, pairs_dir / lname)
        shutil.copyfile(rsrc, pairs_dir / rname)
        pairs_rows.append({"pair_id": pid, "left_file": f"pairs/{lname}", "right_file": f"pairs/{rname}"})
        key_rows.append({"pair_id": pid, "sample": sid, "left_group": left_group, "right_group": right_group})
        row = {"pair_id": pid, "left_file": f"pairs/{lname}", "right_file": f"pairs/{rname}"}
        for side in ("left", "right"):
            for it in ITEMS:
                row[f"{it}_{side}"] = ""
        row.update({"rater": "", "date": "", "notes": ""})
        tpl_rows.append(row)

    def write_csv(path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "blind_pairs.csv", pairs_rows)
    write_csv(out / "answer_key.csv", key_rows)
    write_csv(out / "subjective_template.csv", tpl_rows)
    print(f"OK pairs={len(picked)} out={out}")


if __name__ == "__main__":
    main()
