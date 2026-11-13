# AdaIN 支线：任意参考图风格化（issue #32）

上传**任意内容图 + 任意风格参考图**，输出「保内容、换画风」的结果。

这是与主线并列的**独立支线**，不是同一个模型：

| | 主线（CycleGAN） | 本支线（AdaIN） |
|---|---|---|
| 换的是什么 | **实例外观**：马 → 斑马 | **画风**：照片 → 油画笔触 |
| 输入口 | 一张图 | **两张图**（内容 + 风格参考） |
| 换风格靠什么 | 换**权重文件**（每对单独训练） | 换**风格参考图**（权重不变） |
| 语义 | 会改变物体外观与类别 | 保持不变（马还是马） |

**两类任务的指标不可直接比较**，本支线的产物也不写入 `artifacts/results/`。

---

## 一、跑起来

```bash
cd impl

python cli.py stylize download        # 官方权重：decoder 13.4 MB + VGG 76.4 MB
python cli.py stylize fetch-samples   # 取三张官方风格图（第三方图片不入库）
python cli.py stylize poc             # 内容图 + 风格图 → 出图
python cli.py stylize poc --grid      # 额外输出四联对照（内容/风格/α=1/α=0 重建）
python cli.py stylize poc --content 我的照片.jpg --style 我的画.jpg --alpha 0.7

python cli.py stylize convert --size 256     # PyTorch → OpenVINO IR + 一致性验证
python cli.py stylize quantize --exclude-norm # NNCF PTQ，含四种排除方案的对照
python cli.py stylize bench --size 256        # A / B / C 延迟
```

界面：`python cli.py serve` 后打开 **「05 风格化（支线）」** 页签。
该页支持选内容图 / 上传、选风格图、拖动 α 实时调强度、切换 PyTorch / OV FP32 / OV INT8 后端。

> 本机运行时若 8770 端口报 `WinError 10013`，换端口：`python cli.py serve --port 8780`。

---

## 二、实测（AMD Ryzen 7 7745HX，CPU，OpenVINO 2026.4.1 / NNCF 3.4.0）

### 2.1 延迟与体积

| 尺寸 | A · PyTorch FP32 | B · OpenVINO FP32 | C · OpenVINO INT8 | INT8 体积 |
|---|---|---|---|---|
| 256 | 265.5 ms | 154.4 ms（**1.72×**） | 55.0 ms（**4.83×**） | 26.9 → **7.0 MB** |
| 512 | 1258.6 ms | 690.7 ms（**1.82×**） | 248.5 ms（**5.07×**） | 7.0 MB |

256 尺寸下推荐配置（排除归一化算子）是 **51.4 ms / 5.17×**——比全量化只慢 3.6 ms（6%），
但 PSNR 高出 5 dB（见 2.2）。也就是说**质量几乎白捡**。

### 2.2 INT8 质量：归一化是不是该量化

基准 = OpenVINO FP32 输出（PyTorch→OV 的差异已在 `convert` 里单独验过，max\|Δ\|=7.5e-5）。

| 变体 | 256 · PSNR / SSIM | 512 · PSNR / SSIM | 说明 |
|---|---|---|---|
| `default`（全量化） | 24.05 dB / 0.860 | **16.50 dB / 0.634** | 512 上明显崩 |
| `skip_stats`（排除 ReduceMean+Sqrt） | 28.11 / 0.931 | 26.44 / 0.931 | |
| **`skip_norm_ops`（+Subtract+Divide）** | **29.01 / 0.942** | **28.87 / 0.938** | 推荐 |
| `skip_norm_names`（正则排除 aten::std/mean） | 27.99 / 0.932 | 27.10 / 0.932 | |

三条结论：

1. **归一化确实是量化敏感点**，排除后 256 提升 5.0 dB、512 提升 **12.4 dB**。
2. **分辨率越高，默认量化掉得越狠**（24.05 → 16.50 dB）。校准统计量是在 256 上采的，
   换到 512 特征分布就变了——**校准分辨率必须与推理分辨率一致**，否则误差被放大。
3. **提速几乎不受影响**：排除归一化后 256 仍 5.17×、512 仍 4.76×。

---

## 三、三个必须记住的坑

### 坑 1：AdaIN 里没有 `MVN` 可排除

主线 CycleGAN 用的是 `nn.InstanceNorm`，OpenVINO 会把它融合成**单个 `MVN` 算子**，
所以 NNCF 里 `IgnoredScope(types=["MVN"])` 就能救回 PSNR（12.3 → 29.0 dB）。

AdaIN 的归一化是**手写的** `(c-mean)/std*std_s+mean_s`，转进 IR 后是一堆原语：

```
aten::mean/ReduceMean、aten::sub/Subtract、aten::std/Sqrt、
aten::std/Divide、aten::div/Divide …
```

**没有单个 MVN 节点。** 主线的 `--skip-mvn` 经验不能照搬，得按算子类型或节点名排除
（本支线的 `--exclude-norm` 一次跑四种方案做对照）。

### 坑 2：`ov.save_model` 默认把权重压成 fp16

不传 `compress_to_fp16=False` 时，那份「FP32 基线 IR」存的其实是 fp16，
锁 `INFERENCE_PRECISION_HINT=f32` 也救不回来：实测 max\|Δ\| **0.0146**（0.14% 输出幅度）。
关掉压缩后降到 **7.5e-5**（0.0007%）。**基线不干净，后面所有量化结论都不可信。**

### 坑 3：OpenVINO 默认按 bf16 推理，AdaIN 上比 CycleGAN 更严重

| 尺寸 | 默认（bf16） | 锁 f32 |
|---|---|---|
| 256 | max\|Δ\|=0.525（**5.2%** 输出幅度） | 7.5e-5（0.0007%） |
| 512 | max\|Δ\|=3.753（**38.9%** 输出幅度） | 8.9e-5（0.0009%） |

512 上 bf16 的相对误差接近 39%——如果不锁 `f32`，B 组算出来的「转换差异」根本没有意义。

---

## 四、结构说明

```
stylize/
├── adain.py      网络定义（VGG19 前 4 stage + AdaIN + 9 层 decoder）+ 权重加载 + 前后处理
├── run.py        子命令：download / fetch-samples / poc / convert / quantize / bench
├── .gitignore    权重、风格图、分片都不入库
├── weights/      官方权重（download 生成，不入库）
└── samples/      风格参考图（fetch-samples 生成，不入库）
```

产物统一落 `impl/artifacts/adain/`：`ir/`（IR 与 INT8）、`out/`（出图）、`bench_*.json`、
`quantize_report_*.json`、`convert_report.json`。**不写 `artifacts/results/`。**

结构对齐官方 `naoto0804/pytorch-AdaIN`（MIT）：

- encoder = VGG19 `children[:31]`（到 `relu4_1`）
- decoder = 512→256→(256→256)×3→256→128→128→128→64→64→64→3，3 次 ×2 最近邻上采样，
  **3.51 M 参数 / 13.4 MB**，与官方 `decoder.pth`（14,023,458 B）吻合
- `α` 作为**模型输入**而非常量：否则 trace 后 α 被固化，OV/INT8 后端就无法调强度

---

## 五、隔离约定（写报告时别写反）

- 本支线**不进开题报告**，定位是期末 / 答辩的并列加分项。
- 本支线的指标**不与主线 A/B/C 混表**：AdaIN 语义不变、CycleGAN 改变实例外观，
  两者不是同一任务，FID/LPIPS 放在同一张表里比较没有意义。
- 参考图是「风格」来源，不是「训练数据」：换风格图**不改变**任何权重。
- 论文引用：Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive
  Instance Normalization*, ICCV 2017。
