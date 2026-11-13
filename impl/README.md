# ShiftLite

> 基于 CycleGAN 的实例对象图像迁移，与面向 Intel CPU 的轻量化推理加速。
> 《人工智能导论》课程项目 · 华南师范大学计算机学院 · 4 人小组

一句话：**同一张马图，用三种跑法各生成一次（PyTorch FP32 / OpenVINO FP32 / OpenVINO INT8），然后回答「快了多少」和「质量掉了没有」。**

---

## 一、它做了什么

| 组别 | 含义 | 回答的问题 |
|---|---|---|
| **A** | PyTorch FP32 | 基线，参照系 |
| **B** | OpenVINO FP32 | 只换推理引擎，隔离「运行时收益」 |
| **C** | OpenVINO INT8 | 再压精度，隔离「量化收益」 |

界面提供四个页签：**迁移演示 / 三组性能 / 质量评测 / 实验进度**。

---

## 二、跑起来

### T0/T1 运行与复核交付（2026-11-11）

首次安装、独立副本、缓存路径、完整评测与复核材料导出见
[通用复现指南](../docs/REPRODUCING_T0_T1.md)。Windows 可从仓库根运行
`& .\impl\run-local.ps1 -CliArgs @('serve')`；需先安装依赖并完成 setup。
脚本优先使用仓库 `.venv\Scripts\python.exe`，否则使用 PATH 中的 python；
默认运行副本位于仓库同级 `<仓库名>-runs/local`，不复制历史 artifacts。
可用 `-Python`、`-RunRoot`、`-TorchHome` 分别指定解释器、运行目录和权重缓存。
缓存默认尊重已有 TORCH_HOME，否则放在运行目录的 cache/torch；脚本退出后恢复环境变量。
模型、数据与本机新结果都留在运行副本。运行期间不要同时修改或同步副本代码。

已有 quality-v2 结果可用 `python cli.py review-materials --output <新材料目录>`
导出机器清单、原图/掩码对照和空白 CSV，无需重新推理。
同一目录拒绝混入不同质量 JSON、样本集合或变化后的图片；不改写人工 CSV 或已有模板。
原有 10 张的人工记录入口见
[实验与人工复核归档](../docs/experiments/2026-11-09-7940HX/README.md)。

T1 的 `quality-v2` 在 `deep.structure` 中分别记录输入 `x→a/b/c` 与输出 `a→c`
的 IoU、Dice、共同背景 SSIM，逐指标保存有效/总样本数。共同背景只使用两侧前景
并集外完整的 11×11 窗口，至少 64 个；旧 `background_ssim_ac` 保留为历史填充算法。
每侧取置信度最高目标，输入为 horse、输出为 zebra，检测阈值与掩码阈值均为 0.5。
多实例对应关系未经自动证明，和检测失败一起进入 `results/structure_review/` 的人工待复核清单。
缺失值为 null，不填零、不自动判为生成畸形，也不代签真人复核。

ResNet50 使用权重配套预处理；逐图 Canny 与梯度字段分开。新判别率不得与旧算法混作同一批。
FID 保持原 torchvision 特征实现，不能直接与外部标准 FID 排名比较。
测试入口：在 `impl` 下运行 `python -m unittest discover -s tests -v`。

以下为原始通用安装步骤和历史结果，平台及日期仍按原记录保留。

```bash
# 1. 装依赖（CPU 版 torch，约 250MB）
pip install -r requirements.txt

# 2. 一条命令完成：下模型 → 直转 OpenVINO IR → INT8 量化 → 拉样例图
python cli.py setup

# 3. 三组性能测速（12 轮计时、预热 3 轮、丢弃计时首轮，报均值/标准差/FPS）
python cli.py bench

# 4. 质量与结构保真度量（默认走纯 numpy 结构指标）
python cli.py quality

# 4b. 接真实感知网络（LPIPS/FID/KID/Canny/掩码/判别），首次会下约 600MB 权重
#     正式口径用 --limit 120（testA 全量，方案 §4）；默认 4 张仅供联调
python cli.py quality --deep --limit 120

# 5. E4 层间分工：逐层表示漂移（正式口径同样 --limit 120）
python cli.py layers --limit 120

# 6. 打开可视化界面
python cli.py serve
#    → http://127.0.0.1:8770
```

**换迁移对**：所有子命令都接受 `--pair <id>`，默认是 `horse2zebra`。

```bash
python cli.py setup   --pair zebra2horse     # 斑马 → 马（反向，复用同一份数据集）
python cli.py bench   --pair zebra2horse
python cli.py quality --pair zebra2horse
python cli.py layers  --pair zebra2horse
python cli.py serve                          # 界面里用顶部「迁移对」下拉框切换
```

可用迁移对：`horse2zebra`（默认）、`zebra2horse`、`summer2winter`、`style_monet`，
见 `pairs/` 目录与本文第九节。

首次 `setup` 需要下载 45 MB 权重，网络慢时会自动续传。
Windows 上可先 `set TORCH_HOME=E:\cache\torch`，避免 `--deep` 的权重缓存落到 C 盘。

---

## 三、目录

```
ShiftLite-impl/
├── cli.py                 命令行入口（setup / bench / quality / layers / figures / serve）
├── server.py              本地可视化服务（仅标准库）
├── pairs/                 迁移对清单（每个 pair 一份 JSON：权重源 / 数据集 / 域映射 / 指标开关）
├── core/
│   ├── model.py           CycleGAN ResNet 生成器（与官方结构等价）
│   ├── pairs.py           迁移对加载与路径解析（默认 pair 平铺，其余落 artifacts/<id>/）
│   ├── convert.py         PyTorch 直转 OpenVINO IR + 转换正确性验证
│   ├── quantize.py        NNCF 训练后量化（PTQ）
│   ├── backends.py        A / B / C 三组推理后端
│   ├── metrics.py         SSIM / PSNR / MAE / Dice（结构指标）
│   ├── metrics_deep.py    LPIPS / FID / KID / 掩码 / 判别 / Canny（真实感知指标）
│   ├── layers.py          E4 逐层表示漂移（前向钩子 + NNCF 模拟量化）
│   └── utils.py           图像读写、预处理、计时
├── web/                   可视化界面（纯原生，无外部依赖）
└── artifacts/
    ├── models/            权重、FP32 IR、INT8 IR（默认 pair）
    ├── calib/             量化校准集（trainA，与评测集隔离）+ manifest.json
    ├── samples/           演示用样例图（输入域 / 参照域各几张）
    ├── eval/input|ref/    正式评测集（按角色分，不按 zip 的 testA/testB 分）
    ├── results/           benchmark.json / quality.json / layers.json（默认 pair）
    └── <pair-id>/         其余迁移对的同上结构，互不覆盖
```

> 评测集目录按**角色**命名（`input` = 生成批、`ref` = FID/KID 参照批）而不是沿用 zip 的
> `testA`/`testB`。原因很实际：`zebra2horse` 的输入域在 zip 的 `testB` 里，若目录仍叫
> `testA`，就会出现「testA 里装着斑马」这种自相矛盾，看目录名就会把方向搞反。

---

## 四、⚠️ 两个必须知道的坑（都是实测踩到的）

### 坑 1：OpenVINO 默认按 bfloat16 推理

**OpenVINO CPU 插件在支持的 CPU 上默认按 `bfloat16` 推理。**

实测（随机权重同结构网络）：

| 配置 | 与 PyTorch 的最大逐像素差 |
|---|---|
| OpenVINO 默认 | **0.0371（约 2%）** |
| 显式 `INFERENCE_PRECISION_HINT=f32` | **0.0000061** |

后果：如果不显式锁 `f32`，B 组就不是真正的 FP32 基线，`LPIPS(a,b)` 会被误读成"转换有 bug"，
整条归因链（a-b 查转换、b-c 查量化、a-c 查总差异）就断了。

代码里已经在 `core/backends.py` 与 `core/convert.py` 中统一锁定 `f32`，
并在 `bench` 输出的 JSON 中记录该组实际的 `inference_precision`。

### 坑 2：InstanceNorm 不能一起量化

**CycleGAN 生成器里的 InstanceNorm（IR 中表现为 `MVN` 算子）一旦被量化，整网直接崩。**

实测（同一张输入图，以 B 组 FP32 输出为参照）：

| 量化配置 | PSNR(b,c) | SSIM(b,c) | 生成图 |
|---|---|---|---|
| 默认 PTQ（全量化） | **12.3 dB** | 0.17 | 糊成一片纯色 |
| MIXED 预设 | 11.6 dB | 0.18 | 同样崩坏 |
| **排除 MVN** | **29.0 dB** | **0.93** | 正常 |
| MIXED + 排除 MVN | 29.0 dB | 0.93 | 正常 |

结论：`core/quantize.py` 默认 `skip_instance_norm=True`（即 `ignored_scope=types=["MVN"]`）。
要复现崩溃现象可加 `--keep-instance-norm`。

> 这两条正好对应方案 §3.3 的「INT8 质量不可接受时的备选链」——
> 现在备选链的第 2 档（精度控制量化）有了具体的改进手段作为前置：先排除 InstanceNorm。

---

## 五、实测结果

### 5.1 三组性能 —— 报告主口径：Intel i5-1340P（加速比强平台依赖，AMD 作脚注对照）

**主表 · Intel Core i5-1340P（2026-10-26 重测，接 AC 供电 / 平衡电源计划；预热 3 轮 + 计时 12 轮，弃首轮）**

| 组别 | 均值 | 标准差 | FPS | 体积 | 相对 A | 推理期 RSS 峰值 |
|---|---|---|---|---|---|---|
| A · PyTorch FP32 | 506.3 ms | 17.8 | 2.0 | 43.46 MB | 1.00× | 341.2 MB |
| B · OpenVINO FP32 | 431.2 ms | 38.7 | 2.3 | 43.52 MB | 1.17× | 506.0 MB |
| C · OpenVINO INT8 | 202.6 ms | 17.6 | 4.9 | 11.10 MB | **2.50×** | 568.4 MB |

**复现性（同条件下另跑两轮）**：C 相对 A = 2.42× / 2.50× / 2.74×，稳定；
B 相对 A = 1.17× / 1.40× / 1.39×，**在噪声内跨过 1.0，本机不能说 B 快于 A**——
转换到 OpenVINO 本身在这台机器上没有兑现加速，收益来自量化（C）。

**内存（PLAN §4 四项里的第四项，2026-10-26 首次采集）**：进程 RSS 峰值 341 / 506 / 568 MB；
推理期间增量 A ≤33 MB（首轮触达 PyTorch 内存池）、B 与 C ≈ 0 MB。
**内存占用几乎全部发生在模型与运行时的加载，单次推理不新增驻留。**
三组在同一进程内顺序加载，后测组的绝对值含先前运行时的常驻，跨组横向比较看增量（`rss_delta_mb`）。

**同机对照 · 电池供电（2026-10-24，旧主口径，现降为口径对照）**

| 组别 | 均值 | 标准差 | FPS | 体积 | 相对 A |
|---|---|---|---|---|---|
| A · PyTorch FP32 | 732.2 ms | 27.3 | 1.4 | 43.46 MB | 1.00× |
| B · OpenVINO FP32 | 1125.5 ms | 107.0 | 0.9 | 43.52 MB | 0.65× |
| C · OpenVINO INT8 | 521.3 ms | 47.7 | 1.9 | 11.10 MB | 1.40× |

> **口径订正（2026-10-26）**：两次实测的电源计划同为「平衡」，唯一变量是**供电状态**
> （10-24 电池供电，10-26 接 AC）。B 由 1125 → 431 ms（2.6×），C 由 521 → 203 ms（2.6×），
> A 仅由 732 → 506 ms（1.4×）——**OpenVINO 对供电状态的敏感度显著高于 PyTorch**。
> 此前「未排除：平衡电源计划压制」的说法据此改为**已定位：供电状态**。
> 10-24 的 1.40× 是电池供电下的保守值，报告主表改用插电态的 2.50×，
> 但引用任一延迟比仍必须标注平台与供电状态。

**脚注 · AMD Ryzen 7 7745HX（2026-10-23，队友机器）**

| 组别 | 均值 | 标准差 | FPS | 体积 | 相对 A |
|---|---|---|---|---|---|
| A · PyTorch FP32 | 279.8 ms | 9.3 | 3.6 | 43.46 MB | 1.00× |
| B · OpenVINO FP32 | 138.3 ms | 4.9 | 7.2 | 43.52 MB | 2.02× |
| C · OpenVINO INT8 | 36.7 ms | 2.8 | 27.3 | 11.10 MB | 7.63× |

> **口径决定（2026-10-24 拍板，2026-10-26 复核）**：报告性能主口径用 i5-1340P——方案主张面向
> Intel CPU，这是唯一直接支撑该主张的实测；AMD 的 2.02× / 7.63× 只作脚注对照，
> 引用任一延迟比都必须标注平台。**平台无关的数字**：体积降到 **0.255×**
> （43.46 → 11.10 MB）、量化精度代价 SSIM(b,c)=0.8854 / PSNR(b,c)=28.91 dB——两台通用，
> 可以放心写进结论。
> 已排除的解释：转换链路（直转 IR 与 ONNX-IR 同机同条件 850 ms vs 894 ms，无差异）、
> 线程配置（并行度 5.9–7.2）、测试输入（真实图与随机图无关）、沙箱限制。
> AMD 数据的原机重跑复核留作队友待办（该台数字现只活在 README 与 git 历史，
> `benchmark.json` 已被 i5-1340P 口径覆盖）。

### 5.2 转换正确性（E0）

`max|Δ| = 7e-6`，`mean|Δ| = 1e-6`（随机输入，2026-10-23）；
真实样例图输入按灰度级判据 `max = 0.0014 灰度级 / mean = 0.0001 灰度级`（2026-10-24，阈值 ≤5 / ≤0.5）——两种输入下转换均几乎无损。

### 5.3 质量（正式评测：testA 全量 120 张生成批 vs testB 全量 140 张参照批，2026-10-24）

| 指标 | 数值 | 说明 |
|---|---|---|
| SSIM(a,b) / PSNR(a,b) | 1.0000 / 88.39 dB | 转换无损失，符合预期 |
| SSIM(b,c) / PSNR(b,c) | 0.8854 / 28.91 dB | 量化引入可测但可接受的退化 |
| 边缘 SSIM(b,c) | 0.785（梯度幅值版）/ **0.620（Canny 版）** | 结构保留好于纹理；**方案 §4 指定 Canny 版**，报告引用 0.620，梯度幅值版仅作对照 |
| LPIPS(a,b) / LPIPS(b,c) / LPIPS(a,c) | 0.000 / 0.099 / 0.099 | 真实 LPIPS（AlexNet v0.1），120 张逐图均值 |
| FID / KID（B 批 vs 真实斑马） | 80.70 / 0.0225 ± 0.0066 | 两批分别比较（方案 §4）；KID subset=50×100 |
| FID / KID（C 批 vs 真实斑马） | 81.68 / 0.0252 ± 0.0068 | INT8 后分布偏移仅 +0.98 FID |
| 掩码 IoU(a,c) / 背景 SSIM(a,c) | 0.814 / 0.893 | Mask R-CNN 检出 110/120 对 |
| 斑马判别 top1（B 批 / C 批） | 85.00% / 85.83% | ResNet-50 任务判别，量化前后持平（差值在噪声内） |
| 异常样本登记 | 3 例（ssim_bc < 0.80） | `n02381460_200` / `7170` / `7190`，落 `artifacts/results/anomalies/` |

`SSIM(a,c) ≈ SSIM(b,c)`，与"转换≈0 + 量化=全部差异"的归因预期一致。
判别率 B≈C、FID/KID 仅个位数偏移，说明 INT8 量化未伤任务语义；
6 张样例时代的旧数字（FID 232.6、判别 66.7% 等）样本量不合法，一律作废。

### 5.4 E4 层间分工（真实权重，testA 全量 120 张，2026-10-24）

残差块余弦相似度 **0.9994 → 0.9644 沿深度单调下降**，相对 L2 误差 0.035 → 0.269 单调上升；
编码段（down_relu_1..3）稳定在 0.9996，解码头部（head_conv 0.9859）漂移显著。
量化误差沿残差块逐级累积的分工形态在 120 张全量上成立（15 探针层，n=120，校准 300），
`artifacts/results/layers.json`。

---

## 六、指标口径（写报告时别写反）

- **LPIPS(a,b)** —— 转换本身引入的差异，应≈0；偏大说明转换有 bug，不是量化的锅。
- **LPIPS(b,c)** —— 量化本身引入的差异，核心指标。
- **LPIPS(a,c)** —— 原版到终版的总差异，须独立计算，不能假定等于前两项之和。
- **FID / KID** —— 与真实目标域的**特征分布距离**，**不等于"逼真度"**。
- **掩码 IoU** —— 按类名读取类别索引；只能用 COCO 版 Mask R-CNN（VOC 21 类没有 zebra）。
- **FP16** —— i5-1340P 无 AVX512-FP16，CPU 上不加速、只省体积；仅作核显对照。
- **多迁移对** —— 每对模型独立训练，多接一对只证明**系统兼容**，不证明单模型通用。

---

## 七、当前实现状态（如实说明）

| 能力 | 状态 |
|---|---|
| CycleGAN 生成器结构 + 官方权重加载 | ✅ 已实现 |
| PyTorch → OpenVINO IR **直转**（不经 ONNX） | ✅ 已实现，附一致性验证（灰度级判据自动比对） |
| NNCF PTQ INT8 量化 | ✅ 已实现，默认排除 InstanceNorm（否则输出崩溃） |
| A/B/C 三组延迟、FPS、模型体积 | ✅ 实测 |
| SSIM / PSNR / MAE / Dice | ✅ 纯 numpy 实现 |
| **真实 LPIPS 三对归因**（AlexNet + 官方线性权重） | ✅ `quality --deep`，走 `core/metrics_deep.lpips_three_way_real` |
| **FID / KID**（InceptionV3 2048 维特征） | ✅ `quality --deep`，纯 numpy 求 Frechet 距离与多项式核 MMD |
| **Mask R-CNN 实例掩码 IoU + 背景 SSIM** | ✅ `quality --deep`，COCO 权重按类名取 zebra 索引 |
| **ResNet-50 斑马判别准确率** | ✅ `quality --deep` |
| **Canny 边缘 SSIM**（方案指定） | ✅ `quality --deep`，此前 `metrics.edge_map` 的梯度幅值版仍保留对照 |
| **异常样本登记** | ✅ `quality --deep` 按 `ssim_bc` 阈值落 `artifacts/results/anomalies/` |
| E4 层间分工（逐层表示漂移） | ✅ `cli.py layers`，15 个探针层 + 逐层余弦/相对 L2 |
| 界面默认显示的 LPIPS | ⚠️ **仍是代理值**（不带 `--deep` 时），标了 `"note": "proxy"`，不得写进报告 |

界面「实验进度」页会把每一项标成 **已完成 / 部分完成 / 未开始**，不会把没做的说成做了。

### 真实感知指标怎么开

```bash
pip install -r requirements.txt      # 含 torchvision / opencv-python / lpips
export TORCH_HOME=E:/somewhere/torch # 可选：Windows 上避免权重缓存落到 C 盘

python cli.py quality --deep --limit 120   # 正式口径：testA 全量（方案 §4）
```

首次运行会下载约 600 MB 预训练权重（alexnet / inception_v3 / resnet50 / maskrcnn），
之后走缓存。输出落在 `artifacts/results/quality.json` 的 `deep` 字段，同时
`metrics_source` 会标成 `"real"`（不带 `--deep` 是 `"proxy"`）。
FID/KID 参照批自动取 `eval/testB` 全量 140 张真实斑马图。

---

## 八、与小组方案（`PLAN(2).md`）的对应

| 方案条目 | 落点 |
|---|---|
| §3.2 PoC | `cli.py setup` 中的转换正确性验证（写 `convert_report.json`） |
| §3.3 三组对比 | `cli.py bench` → `benchmark.json` |
| §3.3 量化校准集 | `core/quantize.py`，取 `artifacts/calib`（**trainA**，与评测集隔离） |
| §4 评价指标 | `core/metrics.py`（结构指标）+ `core/metrics_deep.py`（真实感知指标）+ 界面「质量评测」页 |
| §3 网络层间分工 | `core/layers.py` + `cli.py layers` → `layers.json` |
| §6 实验 E0–E5 | 界面「实验进度」页 |
| §9 风险对策 | 量化失败时 `--accuracy-control` 走精度控制量化 |

---

## 九、多迁移对（issue #24）

「换权重即换应用」这件事从代码挪到了数据：一个迁移对 = 一份 `pairs/<id>.json`，
描述**权重源、数据集、域映射、指标开关**，`core/pairs.py` 负责加载与路径解析。

### 9.1 四个迁移对

| id | 类型 | 方向（输入 → 输出） | 数据集 | 数据大小 |
|---|---|---|---|---|
| `horse2zebra` | 实例纹理型 | 马 → 斑马（zip `testA` → `testB`） | horse2zebra.zip | 111 MB |
| `zebra2horse` | 同域反向 | 斑马 → 马（zip `testB` → `testA`） | horse2zebra.zip | **0（复用）** |
| `summer2winter` | 全局域型 | 夏季 → 冬季（`testA` → `testB`） | summer2winter_yosemite.zip | 126 MB |
| `style_monet` | 风格型 | 照片 → 莫奈（`testB` → `testA`） | monet2photo.zip | 291 MB |

四个权重文件都是 **45,575,747 字节**（同一 ResnetGenerator 架构，逐 HEAD 请求核实），
所以转换 / 量化链路完全复用，不需要改代码。

### 9.2 方向陷阱：数据集名和权重方向不是一回事

这是实现时最容易错、而且错了看不出来的地方：

- `style_monet.pth` 对应的数据集叫 `monet2photo.zip`，**没有 `style_monet.zip`**；
  该数据集 A 域是莫奈画、B 域是照片，而 `style_monet` 是 **B→A（照片→画）**。
- `zebra2horse` 同理是反向。

如果按数据集名想当然取 `testA` 当输入，就会把油画当输入、把照片当参照，
结论全错且无从察觉。所以：

- manifest 里显式写 `input_dir / ref_dir / calib_dir`，**不从数据集名推断**；
- 校准集取各自 `trainX`，与 `testX` 评测集隔离，避免数据泄漏；
- 评测集目录按角色命名（`eval/input`、`eval/ref`），不沿用 zip 的 `testA/testB`；
- `zebra2horse` 的 `setup` 日志会打印
  `评测集 输入批 140 张（testB）/ 参照批 120 张（testA）`，方向可当场核对。

### 9.3 路径策略

默认 pair（`horse2zebra`）保持既有平铺路径，已入库的
`artifacts/results/{benchmark,quality,layers}.json` 零改动；其余 pair 落
`artifacts/<pair-id>/`，互不覆盖。这样切换 pair 不会污染别的 pair 的证据。

> 顺带一个真实教训：`bench` 会**直接覆盖**已入库的 `benchmark.json`。
> 我在跑第一次回归时就把 i5-1340P 的那份覆盖了，用 `git checkout` 还原的。
> 这也是路径隔离必须做的实证理由。

### 9.4 `zebra2horse` 实测（本机 AMD Ryzen 7 7745HX）

| 组 | 延迟均值 | 标准差 | FPS | 体积 | 相对 A |
|---|---|---|---|---|---|
| A · PyTorch FP32 | 320.7 ms | 25.4 | 3.1 | 43.46 MB | 1.00× |
| B · OpenVINO FP32 | 188.3 ms | 7.2 | 5.3 | 43.52 MB | 1.70× |
| C · OpenVINO INT8 | 54.6 ms | 7.5 | 18.3 | 11.10 MB | 5.87× |

质量（proxy 指标，6 张）：`ssim_ab ≈ 1.0000`、`psnr_ab ≈ 82–90 dB`（转换无损）；
`ssim_bc 0.855–0.936`、`psnr_bc 28.1–30.6 dB`（量化有可测退化）。

**方向人工核对**：输入是斑马，A/B/C 三组输出的黑白条纹均消失、体色变为棕褐，
形态更接近马 —— 方向正确，无反向生成。

> 注意：绝对值不能与默认 pair 的 **i5-1340P** 数据同表比较（不同机器）。
> 同机回归里 horse2zebra 是 A 254.3 / B 129.6 / C 35.9 ms（本机 AMD 实测，
> 未覆盖仓库里那份 i5 基线），本对整体慢约 25%，属同一台机器上的负载波动，
> 不宜解读为模型差异。

**逐图上更重要的是"谁失效"**：`figures --pair zebra2horse` 出的 E1 拼图
（`artifacts/zebra2horse/figures/fig_e1_samples.png`）里共 3 张，其中

| 样本 | A（PyTorch FP32） | B / C | 判定 |
|---|---|---|---|
| `n02391049_1000` | 画面被棕褐色块状重写，动物结构坏掉 | 与 A 几乎一致（`SSIM(b,c)=0.9358`） | **模型固有失效**，不能算到量化头上 |
| `n02391049_100` | 斑马 → 棕褐色马，效果正常 | 一致（`0.8741`） | 正常出图 |
| `n02391049_10100` | 整幅发灰、几乎看不清 | 一致（`0.8553`） | **模型固有失效**（雾天/逆光场景） |

三张里两张是**纯 PyTorch 基线就已经失效**——这正是方案反复强调的
"要区分模型固有失效模式与量化引入的退化"。若只看 `SSIM(b,c)` 就下结论，
会把 CycleGAN 本身的失效误判成量化的锅。

### 9.5 E4 跨类型层间对比（同机、同一套探针层、同样 120 张）

| 迁移对 | 层数 | 平均余弦 | 漂移最大层 | 该层余弦 | 漂移最小层 | 残差块首→尾 |
|---|---|---|---|---|---|---|
| `horse2zebra`（n=120） | 15 | 0.9906 | `resblock_09` | **0.9644** | `down_relu_2`（0.9996） | 0.9994 → 0.9644 |
| `zebra2horse`（n=120） | 15 | 0.9900 | `resblock_09` | **0.9649** | `down_relu_3`（0.9998） | 0.9995 → 0.9649 |

两对的漂移**形态几乎重合**：`down_*` 几乎不动（≈0.9997），漂移集中在
`resblock_06`–`resblock_09` 并随深度单调累积，最大漂移都落在 `resblock_09`，
两者该层余弦只差 **0.0005**。

结论：**漂移是架构与量化本身的属性，与迁移方向无关**——把马改成斑马、还是把斑马改回马，
被量化损伤的都是同样那几层。这条对报告的意义是：E4 的"层间分工"结论具有方向不变性，
不必为每个迁移对单独重做一遍分析才能外推。

### 9.6 已知限制

- `summer2winter` / `style_monet` 的清单已就位，但数据集（126 MB / 291 MB）
  在本机跨境链路下下载极慢（实测约 40–70 KB/s），**尚未跑通到 benchmark 阶段**；
  界面里它们的下拉项会显示「未就绪」。
- 新迁移对的 `quality` 目前只有 **proxy** 指标（`metrics_source: "proxy"`），
  本机 venv 缺 `torchvision`，跑不了 `--deep`；接入真实感知指标前不得进入正式对比表。
- 深指标字段名已按参照域参数化：FID/KID 键为 `<批>_vs_real_<ref_prefix>`
  （斑马对仍是 `b_vs_real_zebra`，反向对会写成 `b_vs_real_horse`），
  类别判别键改为 `class_discrimination.{target, by_batch}`；
  界面同时兼容旧键 `zebra_discrimination`，已入库的 horse2zebra `quality.json` 无需重跑。
