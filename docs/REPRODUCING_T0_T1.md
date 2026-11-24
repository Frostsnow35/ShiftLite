# T0/T1 运行、评测与人工复核复现指南

本指南从仓库根目录开始。主链为官方 horse2zebra 权重、256×256、CPU 上的 A=PyTorch FP32 / B=OpenVINO FP32 / C=OpenVINO INT8。校准 trainA 300 张；正式质量评测 testA 120 张、testB 140 张。不要用 AMD 性能取代项目的 Intel 主表。

## 1 环境与安装

已验证组合：Windows 11 x64、Python 3.12.4、torch 2.14.1+cpu、torchvision 0.29.1+cpu、OpenVINO 2026.4.1、NNCF 3.4.0。完整版本见 `impl/requirements-cpu-win-py312.lock.txt`，这是本轮 Windows CPU 环境锁。其他 Python、OS 或硬件组合需独立验证。NNCF 3.4.0 官方推荐的 torch 2.13.0 / OpenVINO 2026.4.0 与本轮复现组合不同；见 [官方版本表](https://github.com/openvinotoolkit/nncf/blob/v3.4.0/docs/Installation.md)。

安装 Python 3.12 x64 后，在 PowerShell 进入你自己的仓库根目录执行：

```powershell
py -3.12 -m venv .venv
$py = (Resolve-Path -LiteralPath '.\.venv\Scripts\python.exe').Path
& $py -m pip install -r .\impl\requirements-cpu-win-py312.lock.txt
& $py -m pip check
& $py -c 'import torch, torchvision, openvino, nncf, lpips, cv2; print(torch.__version__, torchvision.__version__); print(openvino.Core().available_devices)'
```

CPU 应出现在 OpenVINO 设备列表中。torch CUDA 为 None 是预期结果；本实验不用 NVIDIA GPU。初次下载需要网络和磁盘空间，模型与数据由 setup 准备，感知网络权重在第一次 `quality --deep` 时下载。安装失败先查具体日志，不在原有 base 环境中反复覆盖依赖。

## 2 建立独立运行副本

下面从实际仓库位置计算目录，不要求 D 盘或特定用户名。该脚本是 PowerShell 入口；Linux/macOS 可在自行准备的独立源码副本内直接调用同一 Python CLI，但本轮没有验证这些平台。

Windows PowerShell 5.1 使用带 UTF-8 BOM 的启动脚本。在仓库根目录执行下列命令可单独检查文件解析。若出现 `PSSecurityException`，用 `Get-ExecutionPolicy -List` 查看各级策略。对本地可信仓库，可用 `powershell.exe -NoProfile -ExecutionPolicy RemoteSigned` 启动仅对该进程生效的会话，在新窗口重新设置 `$py`。组织组策略优先。

```powershell
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
[void][Management.Automation.Language.Parser]::ParseFile((Resolve-Path .\impl\run-local.ps1).Path, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors -join [Environment]::NewLine) }
```

```powershell
$repo = (Get-Location).Path
$run = Join-Path (Split-Path -Parent $repo) ((Split-Path -Leaf $repo) + '-runs\local')
$cache = Join-Path $run 'cache\torch'
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('setup','--subset','300')
if ($LASTEXITCODE -ne 0) { throw 'setup 失败，先检查输出，勿继续宣布实验通过。' }
```

脚本仅复制源码，不复制仓库历史 artifacts。实际模型、数据、结果位于 `$run/impl/artifacts`。默认运行目录也可省略；有现成解释器时用 `-Python` 指向它。每次调用应使用相同的 RunRoot，避免看错实验。停止正在运行的服务后再同步代码，不要在质量评测中同步。

启动器同步 `pairs/*.json` 和 AdaIN 程序。每次验证新版本都选一个新 RunRoot，在同一轮实验中复用它。11 月 9 日及 11 日的历史运行目录用于查阅当时的结果。

setup 完成后检查 `models/convert_report.json` 中 pass=true、same_shape=true，真实图转换误差均值≤0.5、最大≤5 灰度级。默认校准集为 trainA 300 张，默认排除 MVN。当前 `eval/manifest.json` 按角色记录来源与实际写入数，图片位于 `eval/input`、`eval/ref`。默认对的 `input.from/written` 为 testA / 120，`ref.from/written` 为 testB / 140。反向 zebra2horse 的这两组值为 testB / 140、testA / 120，对应清单位于 `artifacts/zebra2horse/eval/`。旧归档保留原字段。

## 3 启动与测试

```powershell
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('serve')
```

终端保持运行，访问 http://127.0.0.1:8770。选择同一张 horse 图，依次验证 A/B/C，均应生成 256×256 图片。按 Ctrl+C 停止服务后再运行实验。只看网页或 --help 不等于推理验收通过。

无需下载模型的代码检查：

```powershell
Push-Location -LiteralPath .\impl
try {
    & $py -B -X utf8 -m unittest discover -s tests -v
    node .\tests\test_quality_ui.cjs
    node --check .\web\app.js
} finally { Pop-Location }
git diff --check
```

Node 只用于前端契约/语法检查，Web 服务运行无需 Node。测试和运行的 Python 应一致。

### AdaIN 联调

**第一档：入口和无权重响应。** 在新的 RunRoot 中把 `@('stylize','bench','--help')` 传给启动脚本，检查输出含 `--size`。进入该副本的 impl，执行 `& $py -B -c 'import stylize.adain'`。启动 serve 后，GET `/api/stylize/state` 应返回 HTTP 200 和 `state.ready=false`。用下面的 JSON 请求 POST `/api/stylize` 应返回 HTTP 409，响应说明缺少权重。

```json
{"content":{"id":"horse_01","source":"samples"},"style":"la_muse","alpha":0.8,"backend":"int8"}
```

**第二档：真实 INT8 出图。** 停止服务。使用已完成默认对 setup 的同一运行副本，确认 `artifacts/samples/horse_01.png` 存在。在仓库根目录运行：

```powershell
foreach ($step in @('download', 'fetch-samples', 'quantize --size 256 --exclude-norm')) {
    $argv = @('stylize') + $step.Split(' ')
    & .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs $argv
    if ($LASTEXITCODE -ne 0) { throw "AdaIN 步骤失败 $step" }
}
```

在运行副本的 impl 下核对 `stylize/weights/{vgg_normalised,decoder}.pth`、`stylize/samples/la_muse.jpg` 及 `artifacts/adain/quantize_report_256.json` 中的 `variants.skip_norm_ops`。该变体成功时应有 `artifacts/adain/ir/adain_256_int8_skip_norm_ops.xml` 和同名 .bin。重启服务，确认 `state.ready=true` 且 `state.backends` 含 int8，然后发送同一个 POST。验收以 HTTP 200、`ok=true`、`backend=int8` 且能从 `output_url` 读取并解码生成图片为准。

同一副本也可验证主线 C 组。在空副本向 `/api/infer` 发送下列请求，缺少输入图时返回 404。停服后准备 `artifacts/samples/horse_01.png` 和 `artifacts/models/horse2zebra_int8.{xml,bin}`，再启动服务。检查 `GET /api/state?pair=horse2zebra` 的 `state.groups` 中 C 组的 `available` 为 true。再次发送请求，应返回 HTTP 200、`group=C`，从本次 `output_url` 能解码出 256×256 PNG。上述资源均可由默认对的 setup 生成。

```json
{"pair":"horse2zebra","sample":"horse_01","source":"samples","group":"C"}
```

复验 E4 时，在空副本依次请求四个迁移对及省略参数的 `/api/state`，检查响应的 `state.layers` 为 null、E4 卡片为 todo。停服后仅将仓库内 `impl/artifacts/results/layers.json` 与 `impl/artifacts/zebra2horse/results/layers.json` 复制到副本中的相同相对位置，再重复请求。正向、反向的卡片应分别对应各自文件，省略参数应等于正向；其余两对仍应为 todo。

## 4 正式实验与表格

```powershell
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('bench','--rounds','12','--warmup','3','--warmup-drop','1')
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('quality','--deep','--limit','120')
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('layers','--limit','120','--subset','300')
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('figures')
```

质量默认仅 4 张、layers 默认仅 6 张，正式记录必须显式指定 120。每条命令检查退出码和日志，不同时执行多个性能/质量/层分析进程。命令会更新运行副本的对应结果；新 session 应使用新的 RunRoot 或先归档。

预期结果：results 下有 benchmark.json、quality.json、layers.json；figures 下有 E1–E4 PNG、quality_table.md 和 structure_table_v2.md。质量表包含 A/C IoU 与 Dice；v2 结构表包含 X/A、X/B、X/C、A/C 的逐指标有效分母。quality.protocol_version=quality-v2、metrics_source=real、deep.errors=[] 表示规定模块已计算；不表示人工认为图片质量合格。partial 必须检查错误，不能用退出码代替完整性检查。

输入保真与输出一致性分开；严格背景与旧填充算法分开；分类器使用权重配套预处理，旧判别率不可混用。检测失败返回 null，不填零；异常样本仍计入可计算指标的全量均值。FID/KID 沿用历史特征实现，不用于跨论文标准 FID 排名。性能、内存及数值可随平台/负载变化，需保存 CPU、供电、依赖和源码信息。

## 5 导出人工复核材料（不重跑模型）

```powershell
$quality = Join-Path $run 'impl\artifacts\results\quality.json'
$materials = Join-Path $run 'review-materials'
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('review-materials','--quality',$quality,'--output',$materials)
```

默认导出当前风险队列：images/、review_manifest.json（质量文件和图片 SHA-256、检测信息、顺序）、detection_failures.csv、空白 human_review_template.csv、相对链接 README。优先原样复制同次评测掩码叠加图；没有叠加图但四侧原图齐全时，明确标记 originals_only_no_masks，不伪造掩码。

只导出原有 10 张，使用仓库中固定样本清单并换一个输出目录：

```powershell
$sampleList = Join-Path $repo 'docs\experiments\2026-11-09-7940HX\legacy_samples.json'
$legacyMaterials = Join-Path $run 'review-original-10'
& .\impl\run-local.ps1 -Python $py -RunRoot $run -TorchHome $cache -CliArgs @('review-materials','--quality',$quality,'--output',$legacyMaterials,'--sample-list',$sampleList)
```

样本在本次 JSON 中缺失、图片来源不完整会报错；4 张联调结果不足以导出原有 10 张。重复导出相同实验不会覆盖已有模板或 human_review.csv；质量文件/选择集合/原图变化则要求新目录。不要直接往已有人工材料目录混入新实验。

复制模板为 human_review.csv 后由真人填写。判定主体是否对应、掩码是否合理、外观与背景是否异常；缺掩码不同于选错主体，检测失败不同于生成畸形。reviewed=已查看，needs_discussion=看过但有疑问，pending=未看。共享整套材料时保留图片和清单；也可以只贴一张 10 图压缩拼图到 PR 评论（GitHub 托管，不占仓库空间），不必传大体积图片包。

## 6 核对既有人工意见与独立复现的区别

[归档入口](experiments/2026-11-09-7940HX/README.md) 保存 2026-11-09 本机结果来源和 2026-11-10 的人工意见。原有 10 张均有第一轮记录，7 条 reviewed、3 条 needs_discussion；署名按用户要求为空。原意见只绑定 image-manifest.json 中的原始图片哈希。独立重跑后即使样本 ID 相同，也不能自动套用这些结论。

只想复核原意见时，看 [PR #31 评论中的压缩拼图](https://github.com/Frostsnow35/ShiftLite/pull/31#issuecomment-6845125203)即可目视判断，无需装模型；拼图是有损缩略图，不能用于哈希核验。需要逐张原图时按上文重跑评测并用 review-materials 本地重建，与 image-manifest.json 的 SHA-256 比对，一致才复用意见，不一致则人工表保持 pending。原图 ZIP 不要求上传。新结果可以另行比较，勿修改旧质量 JSON、删除失败样本或手改均值。

## 7 文件与版本管理

源码、测试、通用指南、依赖锁、小型 JSON/CSV/MD 记录随代码交付；权重、数据、Python 环境、生成 PNG 和缓存按 gitignore 留在本地，原图以 SHA-256 清单锚定、以 PR 评论中的压缩拼图供目视复核。外围日期证据目录不是运行依赖，也不会由 CLI 自动整体生成；保留研究所引用的原始证据即可，无需每位伙伴复制整套安装日志与备份。

AI 参与的 PR 必须在取得实际 PR 编号后按 AGENTS.md 创建 reviews/PR-<编号>.html，双语披露参与范围、自动测试、人工覆盖和未解决问题。图片复核不等于人工代码审查，也不代表 #20 已可关闭。
