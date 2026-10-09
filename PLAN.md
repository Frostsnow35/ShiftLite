# 《人工智能导论》课程项目 — 实施方案（草案）

> 选题方向：图像风格迁移 · 实例对象迁移（马→斑马等）及其推理加速
> 组队：4 人 ｜ 定位：复现 + 小改进 ｜ 交付：报告 + 可运行 Demo + Demo 视频
> 状态：grill 收尾草案，待团队确认后执行

---

## 1. 项目定义（课程阶段①：确定任务）

| 项 | 内容 |
|---|---|
| 题目 | 基于 CycleGAN 的实例对象图像迁移与轻量化推理加速 |
| 项目代号 | 「轻迁 CycleLite」—— 轻量化实例迁移推理系统 |
| 子方向 | C 线「实例对象迁移」（无配对 image-to-image translation），源自指南点名的"更难的实例对象迁移" |
| 实际问题 | 将一种对象的视觉外观迁移到另一种（如马→斑马、苹果→橙子），用通用方法表征并迁移图像风格；同时解决"风格迁移模型推理慢、难部署"的落地痛点 |
| 输入 | 一张内容图（如马的照片、苹果照片） |
| 输出 | 外观/风格迁移后的图（斑马、橙子） |
| 软件形态 | 本地 Web Demo：上传图片 → 选择迁移对 → 实时出图 |

---

## 2. 数据集（课程阶段①：收集数据）

| 用途 | 数据集 | 来源 | 算力 |
|---|---|---|---|
| 训练（可选/加分） | horse2zebra、apple2orange、summer2winter_yosemite 等 | CycleGAN 官方仓库（junyanz/pytorch-CycleGAN-and-pix2pix） | Colab / Kaggle 免费 GPU |
| 推理 / Demo（核心） | 官方预训练生成器 `*.pth` | 伯克利服务器 `http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/<name>.pth`（实测可直下，零花费零训练） | 本地 Intel CPU / Iris Xe |
| 任务型指标 | 马/斑马 二分类器 | 用现成分类模型或自建小样本集 | 本地 |

迁移对规划（多对，体现"产品"通用性）：**主 pair = horse2zebra；对照 pair = apple2orange（或 summer2winter_yosemite）**。至少 2 对。

---

## 3. 技术路线（课程阶段③：设计技术方案）

### 3.1 基线方法
- **CycleGAN**（Zhu et al., ICCV 2017）：无配对双向生成，ResNet 生成器 + PatchGAN 判别器 + 循环一致性损失（cycle consistency）+ identity loss。
- 训练：Colab / Kaggle 免费 T4 跑通一轮，证明"流程跑通"（加分项）。
- 推理：本地加载官方预训练 `latest_net_G.pth`，单图前向出图。

### 3.2 小改进（核心差异化，扣 AI-infra 推理优化）
- 用 **Intel OpenVINO** 将生成器转为 IR 中间表示，做 **INT8 量化**，在本地 Intel CPU / Iris Xe 上加速推理。
- 形成可量化对比：**PyTorch 原始推理 vs OpenVINO 优化推理**。
- 若 INT8 掉点不可接受，退到 FP16 精度。

### 3.3 Demo
- 本地 Web（Gradio 或 Streamlit）：上传图 → 选迁移对 → 出图；录制 Demo 视频。

---

## 4. 评价指标（课程硬性要求，无 ground truth 必须有对策）

| 维度 | 指标 | 说明 |
|---|---|---|
| 推理性能 | 单图延迟(ms)、吞吐(FPS)、模型体积(MB)、内存占用 | PyTorch vs OpenVINO 对比 |
| 生成质量 | LPIPS、FID | 量化前后对比，证"掉点可接受" |
| 任务型（最硬） | 马/斑马 二分类器对生成图的判别准确率 | 证"迁移成功" |
| 主观 | 小样本人工打分问卷 | 兜底 |

---

## 5. 文献综述（课程阶段②）

- **CycleGAN**（Zhu et al., 2017, ICCV）— 核心方法
- **pix2pix**（Isola et al., 2017, CVPR）— 成对对照，理解"配对 vs 无配对"差异
- **Neural Style Transfer**（Gatys et al., 2016, CVPR）— 风格迁移背景
- **Unpaired Image-to-Image Translation**（背景与对比）
- **OpenVINO / INT8 量化**相关文档 — 支撑小改进
- 指南参考链接（CSDN / 掘金 TensorFlow 实现）— 入门实现参考

---

## 6. 实验设计（课程阶段④：实验分析）

| 编号 | 实验 | 产出 |
|---|---|---|
| E1 | 基线推理跑通，出马→斑马样例图 | 样例拼图 |
| E2 | OpenVINO 转换 + INT8，测延迟/吞吐/体积（PyTorch vs OpenVINO） | 性能对比柱状图 |
| E3 | 质量对比 LPIPS / FID（量化前后） | 质量对比表 |
| E4 | 多对泛化：horse2zebra + apple2orange / summer2winter | 多对结果拼图 |
| E5 | 任务型指标：斑马分类器对生成图判别准确率 | 准确率表 |

---

## 7. 四阶段时间表（对齐课程平台截止）

| 阶段 | 时间窗 | 交付 |
|---|---|---|
| 开题报告 | 即日 → 2026-10-21 23:59 | 选题 + 任务定义 + 数据 + 技术路线 + 指标 + 分工初稿 |
| 进展报告 | 2026-10-21 → 11-25 23:59 | 跑通基线推理 + OpenVINO；Colab 训一轮（加分）；出图 |
| 汇报视频 + PPT | 2026-11-25 → 12-23 23:59 | 完整实验图表 + Demo 视频 + 汇报 PPT |
| 期末 + 个人报告 | 2026-12-23 → 2027-01-03 23:59 | 期末课程项目报告 + 个人课程设计报告定稿 |

---

## 8. 4 人分工（建议草案，待定）

| 成员 | 职责 |
|---|---|
| A | 环境搭建 + 模型下载 + 推理流水线跑通（能上云则加训一轮） |
| B | OpenVINO 转换/量化 + 本地 Web Demo |
| C | 文献综述 + 报告撰写（开题/进展/期末三份） |
| D | 实验设计 + 指标计算 + 图表 + Demo 视频 |

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| Iris Xe 无 CUDA，本地不能训 | 训练上 Colab / Kaggle 免费 GPU；核心交付用官方预训练模型本地零训练完成 |
| 风格迁移无 ground truth | FID + LPIPS + 任务型分类器 + 主观问卷多管齐下 |
| INT8 量化掉点 | 测 LPIPS/FID 证可接受；不行退 FP16 |
| 4 人工作量要求高 | 多迁移对 + 多维度指标（性能/质量/任务型）撑 scope |

---

## 10. 创新点表述（避免"纯复现"）

不做单纯 CycleGAN 复现，而是把项目定位为**"通用实例迁移方法 + 面向 Intel 硬件的推理加速落地"**：把风格迁移从"能出图"推进到"能快部署、低延迟"，既差异化，又扣成员 AI-infra 推理优化方向，形成可量化的"PyTorch → OpenVINO"对比证据链。
