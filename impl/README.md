# 轻迁 CycleLite

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

# 2. 一条命令完成：下模型 → 转 ONNX → 转 OpenVINO IR → INT8 量化 → 拉样例图
python cli.py setup

# 3. 三组性能测速（预热 + 多轮，报均值/标准差/FPS）
python cli.py bench --rounds 10

# 4. 质量与结构保真度量
python cli.py quality

# 5. 打开可视化界面
python cli.py serve
#    → http://127.0.0.1:8770
```

首次 `setup` 需要下载 45 MB 权重，网络慢时会自动续传。

---

## 三、目录

```
cyclelite/
├── cli.py                 命令行入口（setup / bench / quality / serve）
├── server.py              本地可视化服务（仅标准库）
├── core/
│   ├── model.py           CycleGAN ResNet 生成器（与官方结构等价）
│   ├── convert.py         PyTorch → ONNX → OpenVINO IR + 转换正确性验证
│   ├── quantize.py        NNCF 训练后量化（PTQ）
│   ├── backends.py        A / B / C 三组推理后端
│   ├── metrics.py         SSIM / 边缘 SSIM / 掩码 IoU / Dice
│   └── utils.py           图像读写、预处理、计时
├── web/                   可视化界面（纯原生，无外部依赖）
└── artifacts/
    ├── models/            权重、ONNX、IR、INT8
    ├── samples/           样例图（校准集也取自这里）
    └── results/           benchmark.json / quality.json
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

## 五、实测结果（AMD Ryzen 7 7745HX，2026-10-23）

> ⚠️ 注意：本机 CPU 是 **AMD Ryzen 7 7745HX**，不是方案里写的 Intel Core i5-1340P。
> 报告里必须如实写实测设备，否则「面向 Intel 硬件」的表述会与数据不符。

### 三组性能（12 轮，预热 3 轮，丢弃首轮）

| 组别 | 均值 | 标准差 | FPS | 体积 | 相对 A |
|---|---|---|---|---|---|
| A · PyTorch FP32 | 279.8 ms | 9.3 | 3.6 | 43.46 MB | 1.00× |
| B · OpenVINO FP32 | 138.3 ms | 4.9 | 7.2 | 43.52 MB | **2.02×** |
| C · OpenVINO INT8 | 36.7 ms | 2.8 | 27.3 | 11.10 MB | **7.63×** |

### 转换正确性（E0）

`max|Δ| = 7e-6`，`mean|Δ| = 1e-6` —— 转换本身几乎无损。

### 质量（6 张 testA 样例均值）

| 指标 | 数值 | 说明 |
|---|---|---|
| SSIM(a,b) / PSNR(a,b) | 1.0000 / ≈87 dB | 转换无损失，符合预期 |
| SSIM(b,c) / PSNR(b,c) | ≈0.92 / ≈29 dB | 量化引入可测但可接受的退化 |
| 边缘 SSIM(b,c) | ≈0.83 | 结构保留好于纹理 |

`SSIM(a,c) ≈ SSIM(b,c)`，与"转换≈0 + 量化=全部差异"的归因预期一致。

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
| PyTorch → ONNX → OpenVINO IR 转换 | ✅ 已实现，附一致性验证 |
| NNCF PTQ INT8 量化 | ✅ 已实现，默认排除 InstanceNorm（否则输出崩溃） |
| A/B/C 三组延迟、FPS、模型体积 | ✅ 实测 |
| SSIM / 边缘 SSIM / 掩码 IoU / Dice | ✅ 纯 numpy 实现 |
| LPIPS 三对归因 | ⚠️ 界面显示的是**结构相似度代理值**，需接入 torchvision 才能得真实值 |
| FID / KID | ⛔ 未接入（需 Inception 权重与完整测试集） |
| Mask R-CNN 实例掩码 IoU | ⛔ 未接入（需 torchvision + maskrcnn 权重） |
| ResNet-50 斑马判别准确率 | ⛔ 未接入 |
| apple2orange 扩展迁移对 | ⛔ 未接入 |

界面「实验进度」页会把每一项标成 **已完成 / 部分完成 / 未开始**，不会把没做的说成做了。

### 待接入项怎么补

```bash
pip install torchvision          # 提供 ResNet-50 / Mask R-CNN / AlexNet 特征
```

然后在 `core/metrics.py` 中把 `lpips_three_way` 换成基于 AlexNet 特征 + 官方线性权重的真实实现，
并新增 `fid_kid()` 与 `mask_iou_via_maskrcnn()`。这三处是期末报告里 E3 / E5 / E6 的关键。

---

## 八、与小组方案（`PLAN(2).md`）的对应

| 方案条目 | 落点 |
|---|---|
| §3.2 PoC | `cli.py setup` 中的转换正确性验证（写 `convert_report.json`） |
| §3.3 三组对比 | `cli.py bench` → `benchmark.json` |
| §3.3 量化校准集 | `core/quantize.py`，取 `artifacts/samples` 前 N 张 |
| §4 评价指标 | `core/metrics.py` + 界面「质量评测」页 |
| §6 实验 E0–E6 | 界面「实验进度」页 |
| §9 风险对策 | 量化失败时 `--accuracy-control` 走精度控制量化 |
