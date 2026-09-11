项目目标：Mamba-3 SISO + 字节级无损压缩研究。

旧项目源码路径： /media/ivc-17/New Volume/zyh/omnizip
旧项目冻结标签（tag）： byte-backbone-baseline-v1
旧项目冻结提交版本（commit）： a5cd9baa24ed2b3959af5ce4572644d6076c69a9
当前阶段： 清理后的基线迁移（clean baseline migration）

## 项目目录

```text
mamba3-byte-compress/
├── README.md
├── .gitignore
├── train.py                       # 训练入口
├── evaluate.py                    # 文件 BPB 评估入口
├── mamba3_byte_compress/           # 核心 Python 包
│   ├── __init__.py
│   ├── config.py                  # 模型名称和训练配置
│   ├── model.py                   # 模型结构
│   ├── data.py                    # 数据读取逻辑
│   ├── metrics.py                 # 损失和 BPB 计算
│   └── trainer.py                 # 训练、断点续训和权重加载
├── datasets/                      # 本地数据，不提交 Git
│   ├── raw/                       # 原始训练、验证和测试文件
│   └── processed/
│       └── train_windows.npy      # 训练窗口起点
├── runs/                          # 本地实验产物，不提交 Git
│   └── baseline_001/
│       └── checkpoints/
│           └── best.pt            # 从原 checkpoints/ 迁入的权重
├── tests/                         # 模型、数据、训练和评估测试
├── scripts/
│   └── activate.sh                # 环境激活脚本
└── docs/
    └── images/
        └── evaluationProcess.png
```

新增核心代码放入 `mamba3_byte_compress/`，运行入口放在根目录，辅助脚本放入
`scripts/`，文档和插图放入 `docs/`。原 `benchmark.py` 仅有占位说明，已移除，
待实现基准测试时再添加入口。

每次训练使用独立的 `runs/<实验名>/`，训练器会在其中保存 `checkpoints/`、
`train.jsonl`、`validation_update_*.json` 和 `training_summary.json`。
评估输出也放在对应的实验目录。已有的 `.conda-envs/` 和缓存目录保持原位。
`datasets/`、`runs/` 已被 `.gitignore` 忽略，克隆仓库后需自行准备数据和权重；
空目录不会随 Git 保存。

## 训练与测试

在项目根目录执行；先准备原始数据，并确保训练窗口起点与训练文件匹配：

```bash
source scripts/activate.sh
python train.py \
  --train-file datasets/raw/train.bin \
  --validation-file datasets/raw/validation.bin \
  --train-starts datasets/processed/train_windows.npy \
  --run-dir runs/baseline_002
```

新的训练使用空目录；继续已有训练时指定原 `--run-dir` 并添加 `--resume`。

运行现有测试：

```bash
source scripts/activate.sh
python -m unittest discover -s tests -v
```

部分测试依赖可用的 CUDA GPU 或冻结的旧项目及其权重，具体以测试输出为准。

## 文件 BPB 评估

使用已配置好的 `mamba3` 环境。在本项目目录执行：

```bash
source scripts/activate.sh
python evaluate.py \
  --test-file datasets/raw/test.bin \
  --checkpoint runs/baseline_001/checkpoints/best.pt \
  --device cuda:0 \
  --output runs/baseline_001/evaluation.json
```

环境实际保存在本项目的 `.conda-envs/mamba3`，通过 Conda 默认环境目录中的
`mamba3` 链接登记，可以在 `conda env list` 中找到，也可以用
`conda activate mamba3` 激活。运行本项目建议使用上面的脚本，它同时设置 CUDA
编译路径和项目缓存目录，不再依赖旧项目的激活脚本或 `/tmp` 环境链接。
默认 Conda 安装位置为 `$HOME/anaconda3`，其他位置可设置 `MAMBA3_CONDA_ROOT`。

需要可用的 CUDA GPU 和 BF16 支持。该激活脚本同时配置 CUDA 编译环境；
在隔离沙盒内看不到 GPU 时，应在允许访问 GPU 的环境中运行。
权重支持本项目固定 Mamba-3 模型的 `best.pt` 和 `latest.pt`，包含模型配置及
`model` 参数字典；不支持只有参数的裸 `state_dict`。仅加载可信的项目 checkpoint，
其训练状态使用 PyTorch pickle 格式保存。

| 参数 | 说明 |
| --- | --- |
| `--test-file` | 必填，原始测试文件；文本和二进制文件均按字节读取 |
| `--checkpoint` | 必填，训练 checkpoint 路径 |
| `--device` | 默认 `cuda:0` |
| `--batch-size` | 默认使用 checkpoint 的 `validation_batch_size`，缺省为 16；显存不足时可调小 |
| `--sequence-length` | 默认使用 checkpoint 的 `sequence_length`，缺省为 2048；改变此值会改变上下文和 BPB |
| `--num-workers` | 数据加载进程数，默认 0 |
| `--output` | 可选，保存 JSON；自动创建父目录，已有结果文件会被覆盖 |

终端输出 JSON，包含 `bpb`、NLL、`num_bytes`、`predicted_bytes`、
`elapsed_seconds`、输入和权重路径、训练步数及运行配置。指定 `--output` 时，
保存相同内容。耗时统计包含评估循环，不包含权重加载。
BF16 推理在不同 batch size 下可能存在微小数值差异；严格对比实验时请固定批大小。

评估使用真实前文预测下一个字节，每块重置上下文，与训练验证口径一致。
每个有效目标计一次，尾部填充不计入损失，文件首字节按均匀分布计 8 bit：

```text
BPB = 总 NLL（自然对数） / (文件字节数 × ln(2))
```

单字节文件为 8 BPB；空文件沿用项目约定返回 0 BPB，并报告 0 字节。
该结果为概率模型的理论编码成本，不包含实际压缩格式开销。
