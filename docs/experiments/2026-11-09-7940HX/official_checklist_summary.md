# 官方复核总清单（2026-11-09 · horse2zebra）

> 来源：`human_review.csv`（人工进度）+ `detection_failures.csv`（机器失败登记）。
> 机器可读版：`official_checklist.csv`（95 行）。机器记录 ≠ 人工结论；`reviewed` ≠ 质量通过。

## 分组统计

| 分组 | 含义 | 数量 |
|---|---|---:|
| P1_recheck_discussion | 已看但仍需讨论，优先独立复看 | 3 |
| P1_first_failure | 未看且本轮检测失败，优先首次查看 | 6 |
| P2_multi_instance | 未看，多实例主体对应待核对 | 79 |
| P3_spotcheck | 已有明确意见，供抽查交叉复核 | 7 |
| 合计 | 复核队列 | 95 |

检测失败：16 个样本 / 34 个失败侧（含原 10 + 新增 6）。

## P1 — 待讨论复看（3）

| 样本 | 失败侧 | 复看重点 |
|---|---|---|
| n02381460_1090 | A,B | 镜中第二只动物在 A/B/C 是否还能辨认 |
| n02381460_1540 | A,B,C | 远景动物条纹是否可辨；天空异常条纹与颗粒感分布 |
| n02381460_3010 | X,A,B,C | 远景剪影是否足以判断；A→B→C 山脊边缘纹理是否加重 |

## P1 — 首次查看（新增检测失败，6）

| 样本 | 失败侧 |
|---|---|
| n02381460_1100 | X,C |
| n02381460_1120 | X |
| n02381460_2890 | X |
| n02381460_3240 | X |
| n02381460_4450 | X |
| n02381460_4660 | X |

## P2 / P3

- P2（79 张）：多实例候选，重点核对「最高置信度掩码是否在各侧选中同一只动物」。逐条见 `official_checklist.csv` 中 `priority=P2_multi_instance`。
- P3（7 张）：n02381460_1260 / 1350 / 1660 / 1750 / 3120 / 440 / 7250，供抽查。

## 使用方式

1. 打开 `official_checklist.csv`，按 `priority` 过滤领取；`comparison_file` 为同目录 PNG（上排 X/A/B/C 原图，下排红色掩码）。
2. 复制 `human_review_template.csv` 为个人表，填 `reviewer/date/conclusion/notes`；未看行保留 `pending`。
3. 回传个人 UTF-8 CSV，汇总时按 `sample` 对照，保留分歧原记录。
