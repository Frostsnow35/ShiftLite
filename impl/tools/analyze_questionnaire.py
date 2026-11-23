"""#15 主观问卷汇总统计（可复用）：均值±SD、Wilcoxon、ICC(2,1)、Krippendorff α。

用法：
  python tools/analyze_questionnaire.py \
    --key ../docs/experiments/questionnaire_T1/answer_key.csv \
    --scores-dir ../human-review/runs/2026-11-23_sim/02_issue15_问卷 \
    --out-dir ../docs/experiments/questionnaire_T1/results

评分人从 `scores_*.csv` 自动发现；四项独立，不合成总分。
零差对在 Wilcoxon 中丢弃（与常见实现一致），输出 `n_used` 以提示功效。
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics as st
from pathlib import Path

DIMS = ["naturalness", "zebra_feature", "pose_structure", "background"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def wilcoxon(x, y):
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i, ties = 0, []
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        ties.append(j - i + 1)
        i = j + 1
    wp = sum(r for r, v in zip(ranks, d) if v > 0)
    wn = sum(r for r, v in zip(ranks, d) if v < 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in ties) / 48.0
    if var <= 0:
        return wp, wn, 1.0, n
    z = (min(wp, wn) - mu + 0.5) / math.sqrt(var)
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return wp, wn, min(1.0, p), n


def icc21(rows, k):
    n = len(rows)
    gm = st.mean([x for r in rows for x in r])
    ms_rows = k * sum((st.mean(r) - gm) ** 2 for r in rows) / (n - 1)
    col_means = [st.mean([r[j] for r in rows]) for j in range(k)]
    ms_cols = n * sum((c - gm) ** 2 for c in col_means) / (k - 1)
    ss_err = sum((r[j] - st.mean(r) - col_means[j] + gm) ** 2 for r in rows for j in range(k))
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return (ms_rows - ms_err) / denom if denom > 0 else float("nan")


def kripp_alpha(units):
    pairs = [(u[i], u[j]) for u in units for i in range(len(u)) for j in range(i + 1, len(u))]
    if not pairs:
        return float("nan")
    do = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
    allv = [v for u in units for v in u]
    n = len(allv)
    de = sum((a - b) ** 2 for i, a in enumerate(allv) for b in allv[i + 1:]) / (n * (n - 1) / 2)
    return 1 - do / de if de > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    key = {r["pair_id"]: (r["left_group"], r["right_group"]) for r in read_csv(args.key)}
    files = sorted(glob.glob(str(Path(args.scores_dir) / "scores_*.csv")))
    if not files:
        raise SystemExit("未找到 scores_*.csv")
    raters = [Path(f).stem.replace("scores_", "") for f in files]

    obs, pairs = {}, []
    for r, f in zip(raters, files):
        cell = {d: {"B": [], "C": []} for d in DIMS}
        for row in read_csv(f):
            pid = row["pair_id"]
            if pid not in pairs:
                pairs.append(pid)
            lg, rg = key[pid]
            for d in DIMS:
                cell[d][lg].append((pid, int(row[d + "_left"])))
                cell[d][rg].append((pid, int(row[d + "_right"])))
        obs[r] = cell
    pairs.sort()

    item_rows = []
    for d in DIMS:
        for g, gname in (("B", "FP32"), ("C", "INT8")):
            vals = [s for r in raters for _, s in obs[r][d][g]]
            item_rows.append({"dimension": d, "group": gname, "n": len(vals),
                              "mean": round(st.mean(vals), 3), "sd": round(st.stdev(vals), 3),
                              "min": min(vals), "max": max(vals),
                              "n_pairs": len(pairs), "n_raters": len(raters)})

    wil_rows = []
    for d in DIMS:
        for scope in ["pooled_pair_mean"] + raters:
            if scope == "pooled_pair_mean":
                xs, ys = [], []
                for pid in pairs:
                    xb = [s for r in raters for p, s in obs[r][d]["B"] if p == pid]
                    xc = [s for r in raters for p, s in obs[r][d]["C"] if p == pid]
                    xs.append(st.mean(xb)); ys.append(st.mean(xc))
            else:
                xs = [s for p, s in obs[scope][d]["B"]]
                ys = [s for p, s in obs[scope][d]["C"]]
            wp, wn, p, n = wilcoxon(xs, ys)
            diff = [a - b for a, b in zip(xs, ys)]
            wil_rows.append({"dimension": d, "scope": scope, "n_pairs": len(xs), "n_used": n,
                             "mean_diff_B_minus_C": round(st.mean(diff), 3),
                             "median_diff": round(st.median(diff), 3),
                             "n_B_better": sum(1 for v in diff if v > 0),
                             "n_C_better": sum(1 for v in diff if v < 0),
                             "W_plus": round(wp, 2), "W_minus": round(wn, 2),
                             "p_two_sided": round(p, 4)})

    cons_rows = []
    for d in DIMS:
        for g, gname in (("B", "FP32"), ("C", "INT8")):
            m = [[next(s for p, s in obs[r][d][g] if p == pid) for r in raters] for pid in pairs]
            cons_rows.append({"metric": "ICC(2,1)", "dimension": d, "group": gname,
                              "n_pairs": len(pairs), "n_raters": len(raters),
                              "value": round(icc21(m, len(raters)), 3)})
    for d in DIMS:
        for g, gname in (("B", "FP32"), ("C", "INT8")):
            units = [[next(s for p, s in obs[r][d][g] if p == pid) for r in raters] for pid in pairs]
            cons_rows.append({"metric": "Krippendorff_alpha_interval", "dimension": d,
                              "group": gname, "n_pairs": len(pairs), "n_raters": len(raters),
                              "value": round(kripp_alpha(units), 3)})

    out = Path(args.out_dir)
    write_csv(out / "item_stats.csv", item_rows)
    write_csv(out / "wilcoxon_table.csv", wil_rows)
    write_csv(out / "rater_consistency.csv", cons_rows)
    print(f"OK pairs={len(pairs)} raters={len(raters)} obs={8 * len(pairs) * len(raters)}")


if __name__ == "__main__":
    main()
