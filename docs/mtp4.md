# MTP-4 概率预测实验

MTP（多 token 预测）在同一时刻的 Mamba-3 隐藏状态上，分别预测后续第 1 至第 4 个字节。模型沿用现有的 2 层 Mamba-3 SISO 骨干，四个预测头均为无偏置的 352→256 线性层。`forward` 返回形状为 `[batch, 2048, 4, 256]` 的预测分数；第三维按预测距离排列，最后一维对应 256 个字节取值。评估时将预测分数转换为概率，读取真实字节对应的概率计算 BPB。模型共有 3,725,772 个可训练参数，原 2 层 baseline 为 3,455,436 个。

训练沿用原始字节文件、训练窗口索引、优化器和 50,000 步配置。现有读取方式为每个样本读取连续 2049 个字节，前 2048 个作为输入。因此一个完整窗口内，四个预测头分别有 2048、2047、2046、2045 个有效目标；窗口末尾缺少的未来字节不计入损失。各头先独立计算平均交叉熵，再等权平均。训练日志记录 `loss1`、`loss2`、`loss3`、`loss4`、`total_loss`，验证记录四头 BPB，并以四头平均 BPB 选出最佳权重。

## 准备数据

现有 `datasets/raw/enwik9` 的前 100,000,000 字节为历史训练集，接下来的 10,000,000 字节为历史验证集。仓库中的训练窗口索引对应前 100 MB。若切分文件尚不存在，在项目根目录运行：

```bash
dd if=datasets/raw/enwik9 of=datasets/processed/enwik9_train_100m.bin bs=1000000 count=100 status=progress
dd if=datasets/raw/enwik9 of=datasets/processed/enwik9_validation_10m.bin bs=1000000 skip=100 count=10 status=progress
sha256sum datasets/processed/enwik9_train_100m.bin datasets/processed/enwik9_validation_10m.bin datasets/processed/train_windows.npy
```

三个文件的 SHA-256 依次应为：

```text
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8
65de6e0008a23a6921af4cfd79ddba1c62b6e38b886f97a4f1e1afdd81c5ceef
df1f06dbb386c052afc27010b86e1d089fda3811f338fa0023d2f96f3b29e197
```

## 训练

在支持 BF16 的 CUDA 设备上激活项目环境，然后运行：

```bash
conda activate mamba3
python train_mtp.py \
  --train-file datasets/processed/enwik9_train_100m.bin \
  --validation-file datasets/processed/enwik9_validation_10m.bin \
  --train-starts datasets/processed/train_windows.npy
```

默认全局 batch 为 32，micro batch 为 32，序列长度为 2048。显存不足时可加 `--micro-batch-size 8`，训练器会累积四个 micro batch，保持全局 batch 为 32。恢复同一实验时指定其目录：

```bash
python train_mtp.py \
  --train-file datasets/processed/enwik9_train_100m.bin \
  --validation-file datasets/processed/enwik9_validation_10m.bin \
  --train-starts datasets/processed/train_windows.npy \
  --run-dir experiments/train/<本次实验目录> --resume
```

每次新训练生成 `experiments/train/<时间戳>_mamba3_siso_3m_mtp4/`。其中 `experiment_data.json` 保存训练损失和验证 BPB，`experiment_summary.json` 保存参数量、数据指纹、batch、累计训练时间及训练进程的峰值 PyTorch 显存分配量；`checkpoints/best.pt` 和 `latest.pt` 分别为最佳与最新权重。训练时间包含该实验的训练、验证、保存和记录过程，恢复训练后累计。显存值单位为十进制 GB，不包含 CUDA 上下文和其他进程。

## 概率评估与对比

在与训练相同的 10 MB 验证集上比较已有 2 层 baseline 权重和 MTP 最佳权重：

```bash
python evaluate_mtp.py \
  --test-file datasets/processed/enwik9_validation_10m.bin \
  --baseline-checkpoint checkpoints/best-3M.pt \
  --mtp-checkpoint experiments/train/<本次实验目录>/checkpoints/best.pt \
  --batch-size 16 --sequence-length 2048
```

BPB 是所有有效预测目标上真实字节负对数概率的平均值。Baseline 与 MTP Head1 使用相同目标位置；各头按自身有效目标数归一化。这个概率评估不计入文件首字节的均匀分布代价，因此与原 `evaluate.py` 输出的整文件 BPB 口径略有区别。

对比结果保存于 `experiments/test/<时间戳>_mtp4_comparison/`：`results.json` 包含每个头的 BPB、有效目标数、参数量、训练用时、显存、batch、数据指纹和 checkpoint；`summary.csv` 为如下表格，空白表示 baseline 没有对应的远期预测头。

| 模型 | 参数量 | Head1 BPB | Head2 BPB | Head3 BPB | Head4 BPB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,455,436 | 1.395367 |  |  |  |
| MTP-4 | 3,725,772 | 1.513876 | 2.251538 | 2.845217 | 3.301937 |

本次完整训练使用 RTX 3090、全局 batch 32、micro batch 32 和 50,000 次更新，累计向模型送入 3,276,800,000 个字节位置，相当于按处理量遍历 100 MB 训练集约 32.768 次；这些训练窗口存在重叠，因此该数值不表示独立数据容量。训练耗时 5,939.21 秒，约 98.99 分钟；训练进程的峰值 PyTorch 显存分配量为 5.676 GB。四头平均验证 BPB 在第 50,000 步达到最低值 2.478142，因此最佳权重与最终权重对应同一步。

MTP Head1 比 baseline 高 0.118509 BPB，表明四个任务等权训练后，单步预测能力有所下降。Head2、Head3、Head4 随预测距离增加依次变差，但均明显低于均匀预测的 8 BPB，说明当前隐藏状态仍包含较远未来字节的有效统计信息。完整机器可读结果位于 `experiments/test/20260921_101857_mtp4_comparison/`，训练记录位于 `experiments/train/20260918_175712_mamba3_siso_3m_mtp4/`。

本实验只比较预测概率，不调用 Range Coding，也不改变编码格式、解码过程或 Mamba 状态更新。
