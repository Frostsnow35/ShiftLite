# T0/T1：7940HX 实验与检测失败人工归档

实验日期 2026-11-09；人工第一轮查看 2026-11-10；交接整理 2026-11-11。代码基线 c998232…，另有未提交改动。硬件为 AMD 7940HX，不能替换项目 Intel 主表。

- [人工表](human_review.csv)：原有 10 张实际意见，署名按要求为空。
- [人工小结与待讨论事项](human_review_summary.md)：7 条 reviewed、3 条 needs_discussion；其余 85 张未查看。
- [固定样本顺序](legacy_samples.json)、[对应检测记录](legacy_detections.json)。
- [原图片哈希清单](image-manifest.json)：10 张对照图与质量 JSON 的 SHA-256，用于原图完整性核验；目视复核用 [PR #31 评论中的压缩拼图](https://github.com/Frostsnow35/ShiftLite/pull/31#issuecomment-6845125203)。
- [实验来源与指标](metrics-summary.json)：数据规模、模型哈希、评测时源码哈希及口径。
- [本机质量表（非项目主表）](quality_table.md)、[结构 v2 表](structure_table_v2.md)：2026-11-09 原质量 JSON 生成；2026-11-11 补充 Dice 展示，未重算质量值。
- 源码哈希核对：与最终 PR 代码比对，仅 `impl/cli.py`、`impl/core/figures.py` 不同（追加 review-materials 子命令与 Dice 展示行，不涉及数值计算），其余 17 个文件在换行符归一后一致。
- [通用复现指南](../../REPRODUCING_T0_T1.md)。

## 复核拼图：一张压缩图，不占仓库空间

10 张对照图的完整 ZIP 不进仓库、也不需要上传。目视复核由**一张拼图**承担，已于 2026-11-13 贴在 [PR #31 评论](https://github.com/Frostsnow35/ShiftLite/pull/31#issuecomment-6845125203)：1024×1550、302 KB JPEG，10 格 2 列 × 5 行，样本 ID 与 [固定顺序](legacy_samples.json) 一致，每格含 X/A/B/C 四侧原图与掩码叠加，无空白或错序。图片由 GitHub 托管，不进 Git、不占仓库体积。

生成规格（供后续重做）：每格宽 512 px、标注样本 ID，JPEG 质量 70，整张控制在 500 KB 内，直接拖进 PR 评论框。

生成示例（在装有 Pillow 的环境，输入目录为 ZIP 解压后的 `images/`）：

```python
from PIL import Image, ImageDraw
import json, pathlib
root = pathlib.Path('images')
order = json.loads(pathlib.Path('legacy_samples.json').read_text())['samples']
cell = (512, 300)
sheet = Image.new('RGB', (cell[0] * 2, cell[1] * 5), 'white')
draw = ImageDraw.Draw(sheet)
for i, sid in enumerate(order):
    im = Image.open(root / f'{sid}.png').convert('RGB')
    im.thumbnail((512, 280))
    x, y = (i % 2) * cell[0], (i // 2) * cell[1]
    sheet.paste(im, (x, y + 20))
    draw.text((x + 4, y + 4), sid, fill='black')
sheet.save('review_contact_sheet.jpg', quality=70, optimize=True)
```

拼图是有损缩略图，只用于目视判断，**不能作为哈希核验依据**。原图完整性仍以 `image-manifest.json` 的 SHA-256 为准：取得同批次原图后用 PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <图片路径>` 比对，一致才复用 CSV 中的人工结论。需要逐张原图时，按 [通用复现指南](../../REPRODUCING_T0_T1.md) 完成 quality-v2 评测，再运行 `python cli.py review-materials --sample-list <本目录/legacy_samples.json> --output <新材料目录>` 本地重建；重建图哈希不一致则人工表保持 pending。

`ShiftLite_原有10张人工复核材料_2026-11-11.zip` 仅为提交者本机留档，不是必需交付物。人工 CSV 是「提交者观察记录 + 哈希锚点」，拼图提供目视复核，#20 仍保持未关闭。

## 独立重新生成材料

在完整 quality-v2 评测完成后，调用 `python cli.py review-materials --sample-list <本目录/legacy_samples.json> --output <新材料目录>`。命令需在独立运行副本的 impl 中执行；或参照通用指南通过 run-local.ps1 调用。

新跑结果仅样本 ID 相同，不保证图片完全相同。新材料的人工表默认 pending，不能自动复用旧意见；以各自 JSON/图片哈希识别。程序的 review_queue 与 anomalies.json 都是机器记录，不代表此处人工结论已自动回写。

## 提交边界

此目录的小型 JSON/CSV/MD 随代码交付；原图 ZIP 与拼图都不入库，拼图由 GitHub 随 PR 评论托管；模型、数据、环境、缓存、安装日志不入库。署名、待讨论项与人工代码审查仍需如实披露。本归档不代表 #20 可直接关闭，正式编号说明见仓库根目录 `reviews/PR-31.html`。
