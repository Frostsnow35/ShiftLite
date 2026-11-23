# 训练复现（独立环节，issue #8）

小规模复现 CycleGAN 训练流程，检验对方法本身的理解。**不进 A/B/C 三组对照，不共用模型文件名。**

## 硬边界（开题三节，勿越）

1. **自训权重不进 A/B/C**：训练引入超参、轮数、数据划分等新变量，会破坏
   「同源权重、只变运行环境与数值精度」的干净对照。主对照回答「运行环境与数值精度的
   影响」，训练复现回答「我们能否复现这套方法」，两张表分开。
2. **训练结论单独成表**：与官方权重对比时如实报告差距，不做美化的并排展示；
   `.pth` 权重不入库（`.gitignore` 已挡），README/报告写获取方式即可。

## 文件

| 文件 | 说明 |
|---|---|
| `train.py` | CycleGAN 训练（G=7 resblock，D=70×70 PatchGAN，LSGAN+循环+身份，Adam 2e-4，100+100 线性衰减，图像池 50） |
| `evaluate.py` | 按 E3 口径算 FID/KID（生成批=testA 输出、真实批=testB，复用 `impl/core/metrics_deep`） |
| `config.json` | 超参（batch=1、lr=2e-4、λ_cycle=10、λ_identity=0.5、seed=20261123） |
| `CycleGAN_smoke.ipynb` | Colab 笔记本：先 20 iter 冒烟，再决定跑量 |

## 数据

`horse2zebra.zip`（126 MB，Berkeley CycleGAN 数据集）：解压出 `horse2zebra/{trainA,trainB,testA,testB}`。
仓库 `impl/artifacts/horse2zebra.zip` 已有本地副本；Colab 直接下载
`https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip`。

## 跑法

```bash
# 冒烟（CPU 也能过，验证损失链与 CSV 落盘）
python train.py --data-root /path/to/horse2zebra --out runs/smoke --n-epochs 1 --max-iters 20

# 正式（Colab T4 约数小时/50 epoch 量级，先 20 epoch 看曲线形状再决定跑满）
python train.py --data-root /path/to/horse2zebra --out runs/exp1

# 每 10 epoch 评一次 FID/KID
python evaluate.py --data-root /path/to/horse2zebra --ckpt runs/exp1/G_AB_ep10.pth \
    --curve runs/exp1/fid_kid_curve.csv
```

## 产出（挂 issue #8 验收）

- `runs/<exp>/train_losses.csv`：生成器 GAN、循环一致、身份、判别器损失逐 epoch；
- `runs/<exp>/fid_kid_curve.csv`：若干 epoch 的 FID/KID；
- `runs/<exp>/config_used.json`：实际超参（可复跑）；
- Colab 笔记本 + 本 README：环境、免费 GPU 实际机型、单次训练耗时填在下表。

## 记录表（训练后填写）

| 项 | 值 |
|---|---|
| 免费 GPU 机型 | 待填（Colab T4 16GB 预期） |
| 单 epoch 耗时 | 待填 |
| 总训练时长 | 待填 |
| 与官方权重差距 | 待填（同 testA/testB 口径 FID/KID 对比，单独成表） |

## 边界与已知限制

- 小规模复现：默认 100+100 epoch 为论文口径，免费 Colab 可先 20–50 epoch 冒烟；
  曲线形状（G/D 博弈、cyc 下降）比绝对值重要。
- 身份损失 λ_identity=0.5 仅对两域共享内容（马↔斑马）有意义，照论文保留。
- 评测口径与 E3 锁定，**不**用校准集（trainA）当评测集。
