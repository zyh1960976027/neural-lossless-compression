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
│   └── debug_20260911_160000_123456/
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

默认名称和自定义名称都会追加运行开始时的本地时间戳，格式为
`YYYYMMDD_HHMMSS_ffffff`（最后六位是微秒）。例如不指定名称时生成
`best_20260911_160000_123456/`，指定 `--name debug` 时生成
`debug_20260911_160000_123456/`。名称支持中文，但必须是一个目录名，不能含路径分隔符。
重名目录不会被覆盖。输出根目录固定相对于项目位置，不受终端当前目录影响。

原来的 `--output` 参数已移除，请使用自动生成的实验目录。JSON 新增 `test_name`、
`experiment_name` 和 `experiment_dir`，用于定位本次运行的结果。
旧 `runs/baseline_001/debug_evaluation.json` 已迁入 `experiments/test/legacy_debug_*/`；
其原始 JSON 内容保留为历史记录，并补充 CSV 表格。

## 训练产物

训练默认保存到 `experiments/train/mamba3_siso_3m/`，也可以显式为新实验指定目录：

```bash
python train.py \
  --train-file datasets/raw/train.bin \
  --validation-file datasets/raw/validation.bin \
  --train-starts datasets/processed/train_windows.npy \
  --run-dir experiments/train/baseline_002
```

训练日志、验证结果和训练摘要保存在该运行目录，训练权重保存在其 `checkpoints/`
子目录。恢复训练时使用相同的 `--run-dir` 并添加 `--resume`。
根目录 `checkpoints/best.pt` 是现有的默认评估权重，不会自动被新训练覆盖；
评估新训练的模型时通过 `--checkpoint experiments/train/<实验名>/checkpoints/best.pt` 指定。

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
