# 评估指标

激活 `mamba3` 后，在项目根目录运行，性能统计默认开启：

```bash
python evaluate.py \
  --test-file datasets/processed/enwik9_debug_64k.bin \
  --batch-size 1 \
  --name debug
```

程序完成时输出 JSON，并在终端最后显示 BPB、推理速度和峰值显存表格。
JSON 写入标准输出，表格写入标准错误输出，便于原有程序继续解析 JSON。
每次运行自动保存完整 JSON 和 CSV 表格，无需指定输出路径：

```text
checkpoints/
└── best.pt
experiments/
├── test/
│   └── 20260911_160000_debug/
│       ├── results.json           # 全部指标、配置和实验目录
│       └── summary.csv            # BPB、推理速度、峰值显存
└── train/                         # 训练输出
```

CSV 可以用 Excel 或其他表格软件打开。终端也会显示结果保存目录。

| 参数 | 默认值与说明 |
| --- | --- |
| `--test-file` | 必填，待评估的原始字节文件 |
| `--checkpoint` | 项目根目录的 `checkpoints/best.pt`，可指定其他权重 |
| `--device` | `cuda:0` |
| `--sequence-length` | 固定默认 `2048`，不再读取权重中的序列长度；可显式覆盖 |
| `--batch-size` | 权重中的验证批大小，缺省为 `16` |
| `--name` | 权重文件名去掉最后一个扩展名，例如 `best.pt` 对应 `best` |

默认名称和自定义名称前都会添加运行开始时的本地时间戳，格式为
`YYYYMMDD_HHMMSS`。例如不指定名称时生成 `20260911_160000_best/`，指定
`--name debug` 时生成 `20260911_160000_debug/`。名称支持中文，但必须是一个
目录名，不能含路径分隔符。
重名目录不会被覆盖。输出根目录固定相对于项目位置，不受终端当前目录影响。

原来的 `--output` 参数已移除，请使用自动生成的实验目录。JSON 新增 `test_name`、
`experiment_name` 和 `experiment_dir`，用于定位本次运行的结果。
旧 `runs/baseline_001/debug_evaluation.json` 已迁入
`experiments/test/<时间戳>_legacy_debug/`；
其原始 JSON 内容保留为历史记录，并补充 CSV 表格。

## 训练产物

默认训练模型为 `mamba3_siso_4l`：4 层、`d_model=352`，骨干参数量
6,550,072，总参数量 6,730,296。相对 2 层基线，仅增加层数；训练数据、优化器、
学习率、batch、训练步数及评估计算方式保持不变。

每次训练默认保存到独立的
`experiments/train/<时间戳>_mamba3_siso_4l/`。可用 `--name` 指定实验名称；
下面的训练和验证路径需要
替换为基线实验使用的真实文件，并复用相同的训练窗口文件：

```bash
python train.py \
  --train-file datasets/raw/train.bin \
  --validation-file datasets/raw/validation.bin \
  --train-starts datasets/processed/train_windows.npy
```

运行目录只保留一个全过程数据表 `experiment_data.json`（训练与验证指标）、一个
配置及最终结果表 `experiment_summary.json` 和一张收敛趋势图
`bpb_vs_steps.png`；训练权重保存在 `checkpoints/` 子目录。趋势图显示验证 BPB、
最佳点，以及最近五次验证的 BPB 改善量，避免训练 batch 的早期高值压缩纵轴，
便于观察后期瓶颈。

4 层实验从头初始化。恢复同一实验时，用 `--run-dir` 指向该实验目录并添加
`--resume`；恢复指定权重可写为 `--resume /path/to/checkpoint.pt`。不支持用 2 层
权重续训 4 层模型。
根目录 `checkpoints/best.pt` 是现有的默认评估权重，不会自动被新训练覆盖；
评估新训练的模型时通过 `--checkpoint experiments/train/<实验名>/checkpoints/best.pt` 指定。

评估会根据权重中的模型名称和配置，严格加载 2 层基线 `mamba3_siso_3m`
或 4 层模型 `mamba3_siso_4l`。例如在同一测试文件上比较两者：

```bash
python evaluate.py \
  --test-file /path/to/same_test.bin \
  --checkpoint checkpoints/best.pt \
  --batch-size 32 --sequence-length 2048 --name baseline_2l

python evaluate.py \
  --test-file /path/to/same_test.bin \
  --checkpoint experiments/train/<时间戳>_mamba3_siso_4l/checkpoints/best.pt \
  --batch-size 32 --sequence-length 2048 --name capacity_4l
```

对比时显式使用相同的评估 batch 和序列长度；上例的 batch 32 与已有完整
enwik9 评估记录一致。BPB、推理计时、显存统计和结果格式沿用原有实现。

## 指标口径

| 字段 | 含义 |
| --- | --- |
| `bpb` | 每个原始字节的理论编码比特数，越低越好 |
| `elapsed_seconds` | 整个评估循环的耗时，包含数据读取、传输、推理和指标计算 |
| `inference_seconds` | 所有批次模型前向推理的累计秒数 |
| `inference_kb_per_second` | `predicted_bytes / 1000 / inference_seconds`，单位 KB/s，越高越好 |
| `peak_gpu_memory_gb` | 评估期间，所选 GPU 上本进程由 PyTorch 分配的峰值显存，单位 GB |

推理计时在数据传到 GPU 后开始，在模型前向完成后结束，包含 Python 调度以及
首次前向可能触发的编译或初始化。计时前后调用 CUDA 同步，确保统计到 GPU
执行完成的时间。未单独进行预热，所以小文件首次运行的速度可能明显偏低。
模型加载、数据读取、CPU 到 GPU 的传输和 BPB 计算不计入推理时间。
性能统计需要逐批同步，因此整体评估耗时可能比未开启统计时稍高。

速度按有效预测字节计数：排除无需模型预测的文件首字节和尾部填充，
但模型处理填充所花的时间仍包含在计时内。单位采用十进制：1 KB = 1000 字节，
1 GB = 1,000,000,000 字节。空文件和单字节文件没有前向推理，推理耗时和速度均为 0。

每次评估开始前重置峰值计数。显存包含已经驻留的模型参数，以及评估期间的
输入、输出和损失计算张量；不包含 PyTorch 缓存分配器保留但未使用的空间、
CUDA 上下文或外部库自行分配的显存，因此与 `nvidia-smi` 的值可能不同。
内部 CPU 测试中显存返回 `null`，表格显示 `N/A`；正式评估仍要求 CUDA。

训练期间的验证默认不启用性能统计，保留原有指标和执行方式。

参考：[CUDA 同步](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)、
[峰值显存定义](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)。
