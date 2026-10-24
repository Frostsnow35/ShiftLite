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
python cli.py quality --deep

# 5. E4 层间分工：逐层表示漂移
python cli.py layers

# 6. 打开可视化界面
python cli.py serve
#    → http://127.0.0.1:8770
```

首次 `setup` 需要下载 45 MB 权重，网络慢时会自动续传。
Windows 上可先 `set TORCH_HOME=E:\cache\torch`，避免 `--deep` 的权重缓存落到 C 盘。

---

## 三、目录

```
ShiftLite-impl/
├── cli.py                 命令行入口（setup / bench / quality / serve）
├── server.py              本地可视化服务（仅标准库）
├── core/
│   ├── model.py           CycleGAN ResNet 生成器（与官方结构等价）
│   ├── convert.py         PyTorch 直转 OpenVINO IR + 转换正确性验证
│   ├── quantize.py        NNCF 训练后量化（PTQ）
│   ├── backends.py        A / B / C 三组推理后端
│   ├── metrics.py         SSIM / PSNR / MAE / Dice（结构指标）
│   ├── metrics_deep.py    LPIPS / FID / KID / 掩码 / 判别 / Canny（真实感知指标）
│   ├── layers.py          E4 逐层表示漂移（前向钩子 + NNCF 模拟量化）
│   └── utils.py           图像读写、预处理、计时
├── web/                   可视化界面（纯原生，无外部依赖）
└── artifacts/
    ├── models/            权重、FP32 IR、INT8 IR
    ├── calib/             量化校准集（trainA，与评测集隔离）+ manifest.json
    ├── samples/           演示与评测样例图（testA / testB）
    └── results/           benchmark.json / quality.json / layers.json
```

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

### 5.1 三组性能 —— 两台机器，两组数字（加速比强平台依赖）

**① AMD Ryzen 7 7745HX（2026-10-23，队友机器）**

| 组别 | 均值 | 标准差 | FPS | 体积 | 相对 A |
|---|---|---|---|---|---|
| A · PyTorch FP32 | 279.8 ms | 9.3 | 3.6 | 43.46 MB | 1.00× |
| B · OpenVINO FP32 | 138.3 ms | 4.9 | 7.2 | 43.52 MB | **2.02×** |
| C · OpenVINO INT8 | 36.7 ms | 2.8 | 27.3 | 11.10 MB | **7.63×** |

**② Intel Core i5-1340P（2026-10-24，复测；平衡电源计划，全核满载约 1.5–1.9 GHz）**

| 组别 | 均值 | 标准差 | FPS | 体积 | 相对 A |
|---|---|---|---|---|---|
| A · PyTorch FP32 | 732.2 ms | 27.3 | 1.4 | 43.46 MB | 1.00× |
| B · OpenVINO FP32 | 1125.5 ms | 107.0 | 0.9 | 43.52 MB | 0.65× |
| C · OpenVINO INT8 | 521.3 ms | 47.7 | 1.9 | 11.10 MB | 1.40× |

> ⚠️ 结论必须写进报告：**2.02× / 7.63× 只在 ① 上成立，在 ② 上 B 反而慢于 PyTorch、C 仅 1.40×。**
> 已排除的解释：转换链路（直转 IR 与 ONNX-IR 同机同条件 850 ms vs 894 ms，无差异）、
> 线程配置（并行度 5.9–7.2）、测试输入（真实图与随机图无关）。
> 未排除：② 受平衡电源计划压制（无法切高性能验证），且 OpenVINO 对两代平台的优化路径不同。
> **报告引用任一数字都必须标注测试平台**；主张「面向 Intel CPU」时，②才应是主口径。

### 5.2 转换正确性（E0）

`max|Δ| = 7e-6`，`mean|Δ| = 1e-6`（随机输入，2026-10-23）；
真实样例图输入按灰度级判据 `max = 0.0014 灰度级 / mean = 0.0001 灰度级`（2026-10-24，阈值 ≤5 / ≤0.5）——两种输入下转换均几乎无损。

### 5.3 质量（6 张 testA 样例均值）

| 指标 | 数值 | 说明 |
|---|---|---|
| SSIM(a,b) / PSNR(a,b) | 1.0000 / ≈87 dB | 转换无损失，符合预期 |
| SSIM(b,c) / PSNR(b,c) | ≈0.92 / ≈29 dB | 量化引入可测但可接受的退化（2026-10-24 复测 0.9128 / 28.8 dB，**已复现**） |
| 边缘 SSIM(b,c) | 0.831（梯度幅值版）/ 0.648（Canny 版） | 结构保留好于纹理；**方案 §4 指定 Canny 版**，报告引用 0.648，另一个仅作对照 |
| LPIPS(a,b) / LPIPS(b,c) | 0.000 / 0.066 | 真实感知指标（`quality --deep`，2026-10-24） |
| FID(B vs 真实斑马) | 232.6 | ⚠️ 仅 6 生成 vs 3 真实，**无统计意义**，正式值需 testA 120 / testB 140 |

`SSIM(a,c) ≈ SSIM(b,c)`，与"转换≈0 + 量化=全部差异"的归因预期一致。

### 5.4 E4 层间分工（真实权重，2026-10-24）

残差块余弦相似度 **0.9994 → 0.9655 沿深度单调下降**，相对 L2 误差 0.035 → 0.265 单调上升；
编码段（down_relu_1..3）稳定在 0.9996，解码头部（head_conv 0.9887）漂移显著。
量化误差沿残差块逐级累积的分工形态成立，`artifacts/results/layers.json`。

---

## 六、指标口径（写报告时别写反）

- **LPIPS(a,b)** —— 转换本身引入的差异，应≈0；偏大说明转换有 bug，不是量化的锅。
- **LPIPS(b,c)** —— 量化本身引入的差异，核心指标。
- **LPIPS(a,c)** —— 原版到终版的总差异，理论约等于前两项之和。
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

python cli.py quality --deep
```

首次运行会下载约 600 MB 预训练权重（alexnet / inception_v3 / resnet50 / maskrcnn），
之后走缓存。输出落在 `artifacts/results/quality.json` 的 `deep` 字段，同时
`metrics_source` 会标成 `"real"`（不带 `--deep` 是 `"proxy"`）。

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
