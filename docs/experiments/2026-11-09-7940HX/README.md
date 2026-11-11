# T0/T1：7940HX 实验与检测失败人工归档

实验日期 2026-11-09；人工第一轮查看 2026-11-10；交接整理 2026-11-11。代码基线 c998232…，另有未提交改动。硬件为 AMD 7940HX，不能替换项目 Intel 主表。

- [人工表](human_review.csv)：原有 10 张实际意见，署名按要求为空。
- [人工小结与待讨论事项](human_review_summary.md)：7 条 reviewed、3 条 needs_discussion；其余 85 张未查看。
- [固定样本顺序](legacy_samples.json)、[对应检测记录](legacy_detections.json)。
- [原图片哈希及附件名](image-manifest.json)：附件内 images/ 与 CSV 相对路径一致。
- [实验来源与指标](metrics-summary.json)：数据规模、模型哈希、评测时源码哈希及口径。
- [质量主表](quality_table.md)、[结构 v2 表](structure_table_v2.md)：2026-11-09 原质量 JSON 生成；2026-11-11 补充 Dice 展示，未重算质量值。
- [通用复现指南](../../REPRODUCING_T0_T1.md)。

## 获取原图附件

代码仓库不包含生成 PNG。将 image-manifest.json 指定的 ZIP 通过团队共享或 PR 附件提供；当前尚未上传，维护者提交时需补可访问的附件入口。伙伴把附件解压后查看其中 README，或将 images/ 置于此目录旁，CSV 即可按相对路径找到图片。

核查文件 SHA-256：PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <图片路径>`，与 image-manifest.json 中值比较（忽略大小写）。不需要安装模型就能复核原意见。

## 独立重新生成材料

在完整 quality-v2 评测完成后，调用 `python cli.py review-materials --sample-list <本目录/legacy_samples.json> --output <新材料目录>`。命令需在独立运行副本的 impl 中执行；或参照通用指南通过 run-local.ps1 调用。

新跑结果仅样本 ID 相同，不保证图片完全相同。新材料的人工表默认 pending，不能自动复用旧意见；以各自 JSON/图片哈希识别。程序的 review_queue 与 anomalies.json 都是机器记录，不代表此处人工结论已自动回写。

## 提交边界

此目录的小型 JSON/CSV/MD 随代码交付，原图附件另交；模型、数据、环境、缓存、安装日志不入库。署名、待讨论项与人工代码审查仍需如实披露。本归档不代表 #20 可直接关闭，也不替代 PR 编号对应的双语 reviewer HTML。
