# T0/T1：7940HX 实验与检测失败人工归档

实验日期 2026-11-09；人工第一轮查看 2026-11-10；交接整理 2026-11-11。代码基线 c998232…，另有未提交改动。硬件为 AMD 7940HX，不能替换项目 Intel 主表。

- [人工表](human_review.csv)：原有 10 张实际意见，署名按要求为空。
- [人工小结与待讨论事项](human_review_summary.md)：7 条 reviewed、3 条 needs_discussion；其余 85 张未查看。
- [固定样本顺序](legacy_samples.json)、[对应检测记录](legacy_detections.json)。
- [原图片哈希清单](image-manifest.json)：10 张对照图与质量 JSON 的 SHA-256，供本地重建后核验；原图不随仓库交付。
- [实验来源与指标](metrics-summary.json)：数据规模、模型哈希、评测时源码哈希及口径。
- [本机质量表（非项目主表）](quality_table.md)、[结构 v2 表](structure_table_v2.md)：2026-11-09 原质量 JSON 生成；2026-11-11 补充 Dice 展示，未重算质量值。
- 源码哈希核对：与最终 PR 代码比对，仅 `impl/cli.py`、`impl/core/figures.py` 不同（追加 review-materials 子命令与 Dice 展示行，不涉及数值计算），其余 17 个文件在换行符归一后一致。
- [通用复现指南](../../REPRODUCING_T0_T1.md)。

## 原图不随仓库交付：哈希绑定与本地重建

10 张对照图体积较大，不作为交付物，也不要求上传 ZIP。`image-manifest.json` 记录每张图与同批次 `quality.json` 的 SHA-256，人工意见只绑定这些哈希。

- 只复核已有意见：取得同批次原图后，用 PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <图片路径>` 与清单比对（忽略大小写）；哈希一致才复用 CSV 中的人工结论，不一致则按新一批材料重新填写。此路径不需要安装模型。
- 没有原图：按 [通用复现指南](../../REPRODUCING_T0_T1.md) 完成一次完整 quality-v2 评测，再运行 `python cli.py review-materials --sample-list <本目录/legacy_samples.json> --output <新材料目录>` 本地重建对照图。重建图哈希与清单一致才复用意见，否则人工表保持 pending。
- `ShiftLite_原有10张人工复核材料_2026-11-11.zip` 仅为提交者本机留档，不是必需交付物；仓库不提供下载入口，也不需要上传。

因此人工 CSV 是「提交者观察记录 + 哈希锚点」：缺少原图时不能独立目视复核，#20 保持未关闭。

## 独立重新生成材料

在完整 quality-v2 评测完成后，调用 `python cli.py review-materials --sample-list <本目录/legacy_samples.json> --output <新材料目录>`。命令需在独立运行副本的 impl 中执行；或参照通用指南通过 run-local.ps1 调用。

新跑结果仅样本 ID 相同，不保证图片完全相同。新材料的人工表默认 pending，不能自动复用旧意见；以各自 JSON/图片哈希识别。程序的 review_queue 与 anomalies.json 都是机器记录，不代表此处人工结论已自动回写。

## 提交边界

此目录的小型 JSON/CSV/MD 随代码交付，原图不入库也不要求上传；模型、数据、环境、缓存、安装日志不入库。署名、待讨论项与人工代码审查仍需如实披露。本归档不代表 #20 可直接关闭，正式编号说明见仓库根目录 `reviews/PR-31.html`。
