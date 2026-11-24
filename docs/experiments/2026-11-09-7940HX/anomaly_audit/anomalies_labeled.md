# 失真样本审计（#17）

> 机器预标注；不代替人工结论；X 侧检测失败与多实例候选不计为生成失真。

- 失真记录：14
- 来源计数：{'baseline_inherent': 10, 'quantization_introduced': 1, 'quantization_aggravated': 9}
- 四类计数：{'形变': 6, '位置偏移': 3, '背景改写': 12, '结构异常': 11}
- 输入侧(X)检测失败（不计失真）：5
- 多实例候选（未计入）：76

## 一、按来源

| 来源 | 样本数 |
|---|---:|
| 基线固有（A/B FP32 已失败） | 10 |
| 量化引入（仅 C 失败） | 1 |
| 量化加重（A/B 与 C 均失败或 ssim_bc 低） | 9 |
| 仅人工记录 | 0 |
| 其他 | 0 |

## 二、按四类

| 类别 | 次数 | 对应层组 |
|---|---:|---|
| 形变 | 6 | resblock_06..09（深层残差，风格/结构） |
| 位置偏移 | 3 | down_relu_1..3 + mask（浅层结构/位置） |
| 背景改写 | 12 | up_relu_1..2 + head_conv（解码，纹理/背景） |
| 结构异常 | 11 | down_relu_1..3（浅层结构/位置） |

## 三、基线固有（A/B FP32 已失败）

| 样本 | 生成侧失败 | 自动类别 | 证据 | 人工 |
|---|---|---|---|---|
| n02381460_1090 | A,B | 结构异常 | 生成侧失败:A,B; 原因:below_score_threshold; X→C 中心偏移比=0.047; 面积比=0.98; edge_ssim_bc=0.784; LPIPS_bc=0.119; ssim_bc=0.897 | needs_discussion |
| n02381460_1260 | A,B | 结构异常,背景改写 | 生成侧失败:A,B; 原因:below_score_threshold; X→C 中心偏移比=0.001; 面积比=0.99; edge_ssim_bc=0.880; LPIPS_bc=0.050; ssim_bc=0.937 | reviewed |
| n02381460_1350 | A,B | 结构异常,背景改写 | 生成侧失败:A,B; 原因:below_score_threshold; X→C 中心偏移比=0.013; 面积比=0.99; edge_ssim_bc=0.875; LPIPS_bc=0.053; ssim_bc=0.936 | reviewed |
| n02381460_1540 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.731; LPIPS_bc=0.162; ssim_bc=0.846 | needs_discussion |
| n02381460_1660 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.816; LPIPS_bc=0.085; ssim_bc=0.900 | reviewed |
| n02381460_1750 | A,B | 形变,结构异常,背景改写 | 生成侧失败:A,B; 原因:below_score_threshold; X→C 中心偏移比=0.034; 面积比=1.15; edge_ssim_bc=0.670; LPIPS_bc=0.178; ssim_bc=0.847 | reviewed |
| n02381460_3010 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.754; LPIPS_bc=0.139; ssim_bc=0.835 | needs_discussion |
| n02381460_3120 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.888; LPIPS_bc=0.040; ssim_bc=0.945 | reviewed |
| n02381460_440 | A,B,C | 形变,结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.698; LPIPS_bc=0.141; ssim_bc=0.824 | reviewed |
| n02381460_7250 | A,B,C | 形变,结构异常,背景改写 | 生成侧失败:A,B,C; 原因:below_score_threshold,no_target_detection; edge_ssim_bc=0.678; LPIPS_bc=0.190; ssim_bc=0.862 | reviewed |

## 四、量化引入/加重

| 样本 | 生成侧失败 | 自动类别 | 证据 | 人工 |
|---|---|---|---|---|
| n02381460_1100 | C | 位置偏移,结构异常 | 生成侧失败:C; 原因:below_score_threshold; mask_iou_ac=0.728; edge_ssim_bc=0.835; background_ssim_ac=0.924; LPIPS_bc=0.083; ssim_bc=0.914 | pending |
| n02381460_1540 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.731; LPIPS_bc=0.162; ssim_bc=0.846 | needs_discussion |
| n02381460_1660 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.816; LPIPS_bc=0.085; ssim_bc=0.900 | reviewed |
| n02381460_200 | - | 形变,背景改写 | X→C 中心偏移比=0.137; 面积比=1.60; mask_iou_ac=0.882; edge_ssim_bc=0.605; background_ssim_ac=0.770; LPIPS_bc=0.191; ssim_bc=0.768 | pending |
| n02381460_3010 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.754; LPIPS_bc=0.139; ssim_bc=0.835 | needs_discussion |
| n02381460_3120 | A,B,C | 结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.888; LPIPS_bc=0.040; ssim_bc=0.945 | reviewed |
| n02381460_440 | A,B,C | 形变,结构异常,背景改写 | 生成侧失败:A,B,C; 原因:no_target_detection; edge_ssim_bc=0.698; LPIPS_bc=0.141; ssim_bc=0.824 | reviewed |
| n02381460_7170 | - | 位置偏移,形变,背景改写 | X→C 中心偏移比=0.461; 面积比=3.66; mask_iou_ac=0.864; edge_ssim_bc=0.554; background_ssim_ac=0.751; LPIPS_bc=0.299; ssim_bc=0.747 | pending |
| n02381460_7190 | - | 位置偏移,形变,背景改写 | X→C 中心偏移比=1.444; 面积比=1.09; mask_iou_ac=0.919; edge_ssim_bc=0.668; background_ssim_ac=0.799; LPIPS_bc=0.162; ssim_bc=0.799 | pending |
| n02381460_7250 | A,B,C | 形变,结构异常,背景改写 | 生成侧失败:A,B,C; 原因:below_score_threshold,no_target_detection; edge_ssim_bc=0.678; LPIPS_bc=0.190; ssim_bc=0.862 | reviewed |

## 五、仅人工记录

| 样本 | 生成侧失败 | 自动类别 | 证据 | 人工 |
|---|---|---|---|---|

## 六、失真类型 × 层漂移对照

| 类别 | 可能对应层组 | 依据 |
|---|---|---|
| 结构异常 | down_relu_1..3（浅层结构/位置） | 浅层≈0.9996 稳定，结构指标仍较高 |
| 位置偏移 | down_relu_1..3 + mask（浅层结构/位置） | 浅层稳定但掩码 IoU 下降 |
| 形变 | resblock_06..09（深层残差，风格/结构） | 残差块首尾 [0.99937362, 0.96436821]，最大漂移 ('resblock_09', 0.96436821) |
| 背景改写 | up_relu_1..2 + head_conv（解码，纹理/背景） | 解码/输出头漂移显著，对应背景 SSIM |

## 七、输入侧(X)检测失败（不计失真）

| 样本 | 原因 | 对照图 |
|---|---|---|
| n02381460_1120 | detection_failure,multiple_instances_correspondence_unverified | n02381460_1120.png |
| n02381460_2890 | detection_failure | n02381460_2890.png |
| n02381460_3240 | detection_failure | n02381460_3240.png |
| n02381460_4450 | detection_failure | n02381460_4450.png |
| n02381460_4660 | detection_failure,multiple_instances_correspondence_unverified | n02381460_4660.png |

## 八、边界

- 自动类别由检测失败侧、X↔C 检测框变化、阈值与关键词产生，需人工复核后定稿。
- 位置偏移/形变来自检测框中心/面积变化，非光流或真值。
- 主表 ssim/LPIPS 与检测批次可能来自不同机器/日期，证据列已如实保留。
- 多实例候选是“主体对应未验证”，不等于生成失败。
