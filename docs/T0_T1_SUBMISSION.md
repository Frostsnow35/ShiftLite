# T0/T1 提交与附件清单

整理日期：2026-11-11。

## 版本状态

前轮 14 个文件的 T0/T1 实现已在本地提交 `2ae30bc` 中，基于上游 `c998232`。本轮未执行暂存、提交、推送或创建 PR；以下是相对当前 HEAD 仍需提交的工作区文件。已有提交和本轮差异都应包含在最终 PR 范围内，不能只交下表而遗漏依赖的前轮实现。

本轮待提交文件：5 个修改、16 个新增，共 21 个。

## 随代码提交

| 状态 | 文件 | 用途 |
|---|---|---|
| 修改 | [README.md](../README.md) | 新增复现与归档入口 |
| 修改 | [impl/README.md](../impl/README.md) | 通用运行参数、材料命令及人工归档入口 |
| 修改 | [impl/cli.py](../impl/cli.py) | review-materials 命令与参数；纠正转换链路注释 |
| 修改 | [impl/core/figures.py](../impl/core/figures.py) | quality_table.md 并列输出 Dice 和有效分母，兼容旧 JSON 缺失 |
| 修改 | [impl/run-local.ps1](../impl/run-local.ps1) | Python/RunRoot/TorchHome 参数、相对默认路径、目录保护、环境恢复 |
| 新增 | [docs/REPRODUCING_T0_T1.md](../docs/REPRODUCING_T0_T1.md) | 首次安装、独立运行、完整评测与材料交付步骤 |
| 新增 | [docs/T0_T1_SUBMISSION.md](../docs/T0_T1_SUBMISSION.md) | 本提交清单与交付边界 |
| 新增 | [docs/T0_T1_VALIDATION.md](../docs/T0_T1_VALIDATION.md) | 本轮实际验证、日期与覆盖边界 |
| 新增 | [docs/experiments/2026-11-09-7940HX/README.md](../docs/experiments/2026-11-09-7940HX/README.md) | 检测失败/异常人工结论归档入口与附件说明 |
| 新增 | [docs/experiments/2026-11-09-7940HX/human_review.csv](../docs/experiments/2026-11-09-7940HX/human_review.csv) | 原有 10 张实际人工意见，签名留空，7 reviewed/3 needs_discussion |
| 新增 | [docs/experiments/2026-11-09-7940HX/human_review_summary.md](../docs/experiments/2026-11-09-7940HX/human_review_summary.md) | 逐张汇总、共同发现、待讨论与范围限制 |
| 新增 | [docs/experiments/2026-11-09-7940HX/image-manifest.json](../docs/experiments/2026-11-09-7940HX/image-manifest.json) | 10 张对照图与质量 JSON 的 SHA-256，用于原图完整性核验 |
| 新增 | [docs/experiments/2026-11-09-7940HX/legacy_detections.json](../docs/experiments/2026-11-09-7940HX/legacy_detections.json) | 原有 10 张的检测侧、类别、置信度及缺失信息 |
| 新增 | [docs/experiments/2026-11-09-7940HX/legacy_samples.json](../docs/experiments/2026-11-09-7940HX/legacy_samples.json) | 原有 10 张固定导出顺序 |
| 新增 | [docs/experiments/2026-11-09-7940HX/metrics-summary.json](../docs/experiments/2026-11-09-7940HX/metrics-summary.json) | 本机环境、模型/源码/质量哈希与原始指标摘要 |
| 新增 | [docs/experiments/2026-11-09-7940HX/quality_table.md](../docs/experiments/2026-11-09-7940HX/quality_table.md) | 7940HX 本机质量表（非项目主表），补 Dice 展示 |
| 新增 | [docs/experiments/2026-11-09-7940HX/structure_table_v2.md](../docs/experiments/2026-11-09-7940HX/structure_table_v2.md) | 输入/输出结构指标及有效分母表 |
| 新增 | [docs/reviews/t0-t1-reviewer-draft.html](../docs/reviews/t0-t1-reviewer-draft.html) | 历史编写稿，已指向正式编号说明 [reviews/PR-31.html](../reviews/PR-31.html) |
| 新增 | [impl/core/review.py](../impl/core/review.py) | 离线导出清单、图片哈希、模板与失败表，保护既有人工作品 |
| 新增 | [impl/tests/test_quality_table.py](../impl/tests/test_quality_table.py) | 验证 Dice 主表与旧 JSON 缺失语义 |
| 新增 | [impl/tests/test_review_materials.py](../impl/tests/test_review_materials.py) | 验证顺序/哈希/人工表保护/跨实验拒绝/无掩码降级 |

## 已在前轮提交中的内容

包括 impl/.gitignore、README、cli.py、core/metrics.py、metrics_deep.py、structure.py、figures.py、web/app.js、index.html、依赖锁、run-local.ps1 和原有三份测试。README/CLI/figures/启动脚本本轮又有增量，需一并纳入下一次提交。

## 本机留档材料，不作为交付物

| 材料 | 交付方式 | 当前状态 |
|---|---|---|
| 10 张对照图压缩拼图（JPEG，< 200 KB） | 由提交者直接拖进本 PR 评论框，GitHub 托管，不进仓库 | 待提交者生成并贴出；规格见 [归档说明](experiments/2026-11-09-7940HX/README.md) |
| ShiftLite_原有10张人工复核材料_2026-11-11.zip | 提交者本机留档，不要求上传 | 已生成并检查图片哈希和 ZIP CRC |
| 附件SHA256.txt | 与 ZIP 一起本地留档 | 已生成 |
| 本轮 clean-run-status.json、测试日志和最终核验 JSON | 按需附于评审材料 | 本地保存，不是运行依赖 |

## 仅本地保留

| 内容 | 原因 |
|---|---|
| Python 环境、torch 权重缓存 | 按指南安装/下载，体积大且与机器相关 |
| PTH、OpenVINO XML/BIN、数据 ZIP、calib/eval/samples | setup 可重建，按 gitignore 不入库 |
| 全量生成 PNG、旧实验备份、安装日志、重复源码快照 | 研究证据可本地/团队归档，无需全部加入代码仓库 |
| 开题报告与课程文稿 | 项目约定为本地交付物 |

## 复核与后续

1. 人工审查代码和 [验证范围](T0_T1_VALIDATION.md)，确认实际认领范围。
2. 生成一张 10 图压缩拼图并贴到本 PR 评论，供目视复核；原图 ZIP 不用上传，规格见 [归档说明](experiments/2026-11-09-7940HX/README.md)。
3. 如实说明 10 张第一轮已看、3 张有待议项、署名为空、其余 85 张未看；是否处理新增 6 张失败由小组确定。
4. 正式编号说明为 [reviews/PR-31.html](../reviews/PR-31.html)，按 AGENTS.md 随 PR 提交。
5. 不把 #20 自动写为已关闭；讨论项和人工代码审查尚未完成。

本表不包含 git add/commit/push 的自动执行，也未触发 GitHub 评论或状态修改。
