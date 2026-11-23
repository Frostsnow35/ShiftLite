# 主观问卷（#15）材料说明

本目录用于 PLAN §4「主观（兜底）」指标：比较 **FP32(b)** 与 **INT8(c)** 的生成图观感差异。领队拍板：组内 4 人互评、成对盲评、随机左右。

## 文件

- `pairs/`：60 张盲评图（30 对，`pair_XXX_L/R.png`，中性命名，不泄露组别）。
- `blind_pairs.csv`：发给打分人，仅含 `pair_id` 与左右图路径。
- `subjective_template.csv`：打分表，4 项 × 左右两侧 + `rater/date/notes`。
- `answer_key.csv`：`pair_id -> sample, left_group, right_group`，**仅供汇总人使用，打分人请勿查看**。注意本文件已入库（复现留存），且生成种子 `20261120` 公开可推导，盲评有效性依赖打分人自觉。
- `rubric.md`：四项定义与 1–5 锚点、汇总口径。
- `results/`：#15 统计结果（`item_stats.csv`、`wilcoxon_table.csv`、`rater_consistency.csv`、`results.md`、`scores/`）。

> `pairs/` 为本地生成、**不入库**（体积与来源限制）；分发问卷时把 `questionnaire_T1/` 目录打包给打分人，但**剔除 `answer_key.csv`**，或按下方命令在本地重新生成。

## 材料来源与口径

- 盲评图来自本机 2026-11-19 的 `quality --deep --limit 120` 质量运行（`impl/artifacts/results/quality/<sample>_{B,C}.png`），
  样本从 120 张中按固定随机种子 `20261120` 抽取 30 张。
- 这是**主观兜底**，与 #20 的检测失败人工复核（2026-11-09 官方批次）是两件事，不要混用结论。
- 若需要与官方 2026-11-09 批次完全对齐，需用该批次的 B/C 输出重新生成盲评图（当前包内未含该批次 B/C 原图）。

## 使用流程

1. 复制 `subjective_template.csv` 为 `scores_<组员>.csv`。
2. 按 `blind_pairs.csv` 打开左右图，依 `rubric.md` 对四项各打 1–5 分（左右分别打）。
3. 填 `rater/date`；不确定处写 `notes`。
4. 回传 CSR 后，汇总人用 `answer_key.csv` 还原组别，按 `rubric.md` 的汇总口径统计。

## 再生成

```powershell
cd impl
python tools/build_blind_pairs.py --quality-dir artifacts/results/quality --out-dir ..\docs\experiments\questionnaire_T1 --n 30 --seed 20261120
```
