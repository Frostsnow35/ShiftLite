# 数据来源与人工确认（Provenance）

- 本目录的 `human_review_merged.csv` 与 `detection_failures_merged.csv` 来自 2026-11-23 的第二轮人工复核；首轮 2026-11-10 记录保留在 `../human_review.csv` 与 `../human_review_summary.md`。
- 复核人 `问卷人A`–`问卷人D` 为组内 4 位评分人代号；四人分别完成所分样本的查看与记录，真实姓名不写入仓库。
- P3 抽查 7 张沿用 2026-11-10 首轮状态，本轮未重复查看。
- 机器字段（`review_queue.json`、`detection_failures.csv`、质量 JSON）未被改写，人工表只新增状态/结论/备注。
- `reviewed` 不等于质量通过；`needs_discussion` 保留待组内讨论。
- 本批人工结论仅适用于本批图片；独立重跑即使样本 ID 相同，也不能自动套用。
- AI 参与范围：AI 辅助整理表格、合并与统计脚本；人工观察文字由复核人提供/确认。
