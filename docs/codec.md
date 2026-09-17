# 无损编解码

默认使用 **batched-step-v5 + v5**，默认同时处理 128 个独立的 2048-byte 块，可通过 `--batch-size` 增大。
每块首字节使用均匀分布，其余字节根据 Mamba 循环状态预测；循环状态保存该块已经处理的历史信息。
模型结构、权重及训练代码不变。模型在 GPU 上生成 256 个字节取值对应的概率，再交给
Range Coding（区间编码），根据预测概率进行无损压缩。每批块共享一条编码流，同一位置的
有效字节一起处理。v5 通过固定缓冲区和 CUDA Graph 加速推理，解码必须精确恢复原文。

## 使用

```bash
source scripts/activate.sh
python codec.py encode \
  --input /path/to/input.bin --output /path/to/input.m3rc \
  --checkpoint checkpoints/best-3M.pt --batch-size 128 \
  --results /path/to/encode.json
python codec.py decode \
  --input /path/to/input.m3rc --output /path/to/decoded.bin \
  --checkpoint checkpoints/best-3M.pt --results /path/to/decode.json
cmp /path/to/input.bin /path/to/decoded.bin
```

`--batch-size` 默认 128，已取消 128 的固定上限；文件头允许 1..4,294,967,295（uint32）。
实际大小受显存、系统内存和剩余块数限制，仅适用于编码/roundtrip 的 batched-step。
程序不会根据空闲显存自动缩小 batch，编码和解码严格沿用文件头分组参数。
解码从头部读取推理配置与 batch size，不能覆盖。`--device` 默认 `cuda:0`。
本机实验权重是 `checkpoints/best-3M.pt`；省略 checkpoint 时仍使用原项目的
`checkpoints/best.pt`。依赖见 `requirements-codec.txt`。

也可使用 `python codec.py roundtrip --input ... --compressed ... --decoded ...`，
它会执行编解码并逐字节比较。输出路径父目录须存在，已有文件不会被覆盖；
发生异常会删除本次不完整输出。解码只需要压缩文件、同一 checkpoint 和兼容运行环境。

## 批量 state 与独立块

文件按原始顺序切成互不重叠的块。块数为 `ceil(N/2048)`，每块首 byte 成本为 8 bit。
每批实际 B 为 `min(batch_size, 剩余块数)`。GPU 状态按 B 维度隔离，每批开始清零；
各块首 byte 使用均匀分布写入当前 group 的共享 Range 流，无须模型预测。

随后每步输入 `[B]` 个已知 byte，经过 embedding、各层原生 `Mamba3.step`、
残差、MLP、归一化与输出头，输出 `[B,256]` logits。每个完整块最多调用 2047 次。
各层 angle/SSM 状态为 FP32、K/V 为 BF16，原地更新，不重算历史，无 prefill。
整批完成后按块原始顺序写出。新批次清零状态，改变 B 时重新分配缓存。

短块结束后，其槽位后续输入为零，输出被忽略；批内保持 B 不变。
编码与解码使用完全相同的分组、形状、补零规则。
**batch size 是数值协议的一部分，不保证不同 batch size 得到相同概率或码流。**

每步仅进行一次批量输入上传。GPU 将 logits 转换为 FP32 概率后，通过可复用的 pinned-memory
缓冲区一次传回 CPU。Constriction 通过二维概率表批量量化该位置所有活动块的概率，并在一个
group 共享的 Range Encoder/Decoder 中批量处理这些 byte。所有 byte 均保留可编码的非零质量。
使用 eval、inference mode、BF16 autocast、FP32 权重、关闭 TF32 和确定性选项。
v4 路径不使用 CUDA Graph；v5 的推理执行方式见下文。

当前 3M 模型有 11 个 head。已安装 CuTe step 要求外层 stride 按 16 bytes 对齐，
原生预处理的 `[B,11]` 张量与部分投影视图不满足批量调用条件。
`codec_step_adapter.py` 仅将这些中间值复制到对齐存储，保留形状、dtype 和数值；
仍调用原生 step，没有修改第三方安装包，也没有增加 head 或 token。

Python API：

```python
from mamba3_byte_compress.codec import load_predictor
predictor = load_predictor("checkpoints/best-3M.pt", batch_size=128)
predictor.reset(2)
probabilities, _ = predictor.step([65, 66])  # FP32 [2,256]，更新两个独立状态
```

文件编解码会自动 reset。直接调用 step 时，累计 2047 次后必须显式开始新批次。

## v4 文件格式（v5 沿用布局）

整数均为小端序。固定开销 110 bytes，每个 batch group 额外使用 8-byte 帧头：

| 部分 | bytes | 内容 |
| --- | ---: | --- |
| magic | 8 | `M3RC\x04\r\n\x1a` |
| 概率配置编号 | 1 | 4，GPU 概率与 group 共享 Range 流 |
| flags | 1 | 0，保留其他值并拒绝读取 |
| 原始长度 | 8 | uint64，推导块数与尾块长度 |
| batch size | 4 | uint32，范围 1..4,294,967,295 |
| checkpoint 指纹 | 32 | 完整权重文件 SHA256 |
| 环境指纹 | 16 | runtime JSON 的 SHA256 前 16 bytes |
| 头部 CRC32 | 4 | 覆盖前面全部 70 bytes，包括 batch size |
| 每组帧 | 可变 | uint32 payload 长度、uint32 payload CRC32、Range payload |
| 主体 CRC32 | 4 | 覆盖全部帧头与 payload |
| 原文 SHA256 | 32 | 验证最终恢复字节 |

Range payload 为小端 uint32 字序列，包括各流收尾及对齐。每个 batch group 只有一个流；
流内顺序为该组全部块的首 byte，随后按位置依次写入所有仍活动的块。
空文件没有帧；单字节文件只有一个均匀符号帧。
参数不合法、头/主体/帧 CRC 错误、截断、多余主体数据、权重或环境不匹配均拒绝。
解码后的 SHA256 必须与容器一致，否则删除输出。
当前解码会将压缩主体读入内存，未实现随机访问索引或大文件流式解码。

环境指纹使用 `json.dumps(runtime, sort_keys=True, separators=(",", ":"))` 计算，
包括 GPU 型号、PyTorch/CUDA/NumPy/Mamba/Triton/Constriction 版本，以及 step 依赖的
`nvidia-cutlass-dsl`、`quack-kernels`、`cuda-bindings` 版本。详细值保存在结果 JSON。
版本检查与确定性开关不能保证所有第三方内核跨硬件逐位一致，最终 SHA256 是结果验证。

## v5 推理执行优化

`--inference batched-step-v5 --format v5` 使用固定缓冲区及 CUDA Graph。CUDA Graph
预先记录一次模型预测及 GPU 概率生成的运算，后续逐字节重放，减少重复调度。
每层重复使用固定投影张量、对齐存储和输出空间；模型结构、训练权重、计算精度及
Mamba 状态更新内核保持不变。实现位于 `mamba3_byte_compress/codec_graph.py`，不修改安装包。

每个实际批大小拥有独立的循环状态、输入输出缓冲区和图，最多缓存两种批大小，
超过时淘汰最近最少使用的条目。尾批使用实际大小，不填充到最大批大小。
准备时先完成内核编译和预热，再捕获图；准备完成及每批开始均原地清零状态。
输入上传、概率下载和 CPU 区间编码位于图外。图捕获失败会报错，不自动切换路径。

v5 沿用 v4 的容器布局，magic 为 `M3RC\x05\r\n\x1a`，概率配置编号为 5。
其余分块、校验和帧规则相同；旧 v4 文件仍使用旧预测器解码。
编码和解码均要求支持该版本的程序及相同权重、兼容环境。

结果 JSON 新增 `graph_prepare_seconds`（累计图准备耗时，含实际发生的编译和预热）、
`graph_capture_count`、`graph_cache_hits`、`graph_cached_batch_sizes` 和 `graph_enabled`。
未提前准备的普通 CLI 会将图准备计入编解码时间；稳态实验提前准备所有实际批大小，
并单独记录加载、准备、完整进程时间与显存峰值。

完整对照实验：

```bash
python scripts/benchmark_codec_inference.py \
  --output experiments/test/codec_inference_new_run
```

默认沿用 enwik9 `[100000000, 110000000)`、3M 权重及批大小 1024，比较原始 v4、
仅复用缓冲区的 v5、启用图重放的 v5。每端独立进程，三次重复；若基线与图重放
的耗时范围重叠，增加至五次。另测新进程跳过预热的首次运行，保留磁盘编译缓存。
输出中文报告、JSON、CSV、码流及逐字节校验记录。已有完成运行可在同一目录继续汇总；
更改实现或实验配置后应使用新目录。

单独运行消融实验，可在 `benchmark_codec_batched.py` 中指定 `--format v5`
及 `--execution buffered`；默认 `--execution graph`。`--cold` 跳过提前预热。
普通编解码接口仅提供正式图重放路径。

GPU 正确性检查：

```bash
MAMBA_CODEC_GPU_TEST=1 python -m unittest discover -s tests -p test_codec_graph.py -v
```

检查 1024、787 两种批大小的完整 2047 步递推、图与普通执行的概率及状态一致性、
状态清零、批大小切换和缓存淘汰。测试需使用可访问本地 GPU 的运行环境。

## 旧格式兼容

v4 可通过 `--inference batched-step-v4 --format v4` 显式使用，已有 v4 文件自动选择旧预测器解码。

v3 仍可读取，也可通过 `--inference batched-step --format v3` 显式编码。v3 使用 CPU float64
概率生成，每个 2048-byte 块保存独立 Range 流和 8-byte 帧头；其 batch size 同样记录在文件头。

`--inference step` 保留旧单槽位状态推理，`--inference prefix` 保留旧完整历史前向；
这两者默认输出 v2，也可显式指定 `--format v1` 或 `--format v2-framed`。
只指定旧格式而未指定 inference 时自动选择对应推理方式。v5 仅允许 batched-step-v5；v4 仅允许 batched-step-v4，
v3 仅允许 batched-step。
旧码流解码自动识别，不要求额外参数。

旧语义为：文件首 byte 均匀编码，后续每组 2048 个预测目标以前一个 byte 为种子，
相邻组共享种子 byte。step 每 2048 次预测重置，prefix 每次完整前向 `[1,2048]`。
两者概率配置分别为 2、1，不能混用。旧配置的运行环境指纹字段保持不变。

- v2：magic `M3RC\x02\r\n\x1a`，66-byte 紧凑头加 4-byte 头 CRC，
  主体为全文件一个 Range 流，尾部为 4-byte 主体 CRC 和 32-byte 原文 SHA256，固定开销 106 bytes。
  头字段依次为 magic、配置编号、流模式、原始长度、权重指纹、环境指纹；无 batch size。
- v2-framed：同一头尾，流模式为 0，保留首 byte 帧及逐预测块帧；v2 共享流模式为 1。
- v1：magic `M3RC\r\n\x1a\n`、uint32 JSON 头长度、JSON 元数据、首 byte 帧、预测块帧、32-byte 原文 SHA256。
  每帧同样为长度、CRC32、Range payload。

## 指标与正式实验

| 字段 | 计算方式 |
| --- | --- |
| `theoretical_nll_bpb` | `(块数 × ln(256) + 模型预测 NLL) / (N × ln(2))` |
| `actual_bpb` | `完整压缩文件 bytes × 8 / N` |
| `payload_bpb` | `全部 Range payload bytes × 8 / N` |
| `encode_kb_per_second` | `N / 1000 / encode_seconds` |
| `decode_kb_per_second` | `N / 1000 / decode_seconds` |

理论 NLL 在定点概率量化前计算。实际 BPB 还包括流收尾、对齐、帧头与文件头尾。
新的独立块/均匀首 byte 口径与原有整块 forward BPB 评估不同，不要求两者相等。
空文件理论 BPB 为 0，实际/payload BPB 为 null，速度为 0。

端到端函数计时覆盖读取、状态初始化、推理、传输、CPU 概率处理、熵编码、校验、写入及关闭文件。
GPU 在开始和结束时同步；不强制磁盘 fsync，不包含模型加载及额外的原文逐字节比较。
普通 CLI 没有单独预热。正式实验脚本会单独记录模型加载及简单预热，预热后清零状态：

```bash
python scripts/benchmark_codec_batched.py \
  --output experiments/test/codec_batched_1mb_20260915 \
  --checkpoint checkpoints/best-3M.pt
```

单次实验脚本 `benchmark_codec_batched.py` 默认使用 v5，取 enwik9 前 1,000,000 bytes，可通过 `--sample-bytes` 指定长度。
1 MB = 1,000,000 bytes，1 KB = 1000 bytes。
B=128 时有 489 块（488 个完整块及 576-byte 尾块），分为 128、128、128、105 四批。
均匀 byte 数为 489，模型预测 byte 数为 999,511，每端批量模型调用为 8,188 次。
两个独立子进程分别执行一次完整编码与解码，主进程再逐字节及 SHA256 比较。
CPU PyTorch 线程数固定为 1 并记录，使概率量化和 Range Coding 的测量口径稳定。
每次命令只测试一个数据长度和一个 batch size，不自动扫描。保存原文、压缩文件、恢复文件、JSON、CSV 和日志。

例如测试 enwik9 前 **10 MB**、**batch size 1024**：

```bash
python scripts/benchmark_codec_batched.py \
  --sample-bytes 10000000 --batch-size 1024 \
  --checkpoint checkpoints/best-3M.pt \
  --output experiments/test/codec_batched_10mb_b1024_new_run
```

10 MB 对应 4,883 个块（4,882 个完整块及 1,664-byte 尾块），B=1024 分为
1024、1024、1024、1024、787 五批。输出目录必须不存在。
v5 文件需使用支持 v5 的解码器；已有 v4、v3 文件仍可读取。

实验脚本还记录加载、预热、正式编解码三个阶段的 PyTorch CUDA 显存峰值。
`codec_peak_allocated_bytes` 是正式阶段张量分配峰值，`codec_peak_reserved_bytes`
是缓存分配器预留峰值；`run_peak_*` 是三个阶段峰值的最大值。数值以 bytes 保存，
报告用 MiB（1 MiB = 1,048,576 bytes）。每阶段开始重置峰值计数，不调用
`empty_cache`，因此包含常驻模型/状态及已有缓存。统计不含 CUDA 上下文、
非 PyTorch 管理的显存和其他进程，不能视为整卡或整个进程的显存峰值。

## 轻量检查

```bash
python -m unittest discover -s tests -p test_codec_step.py -v
```

覆盖状态独立及批次清零；v3/v4 的 4098-byte、B=2 独立编解码（含空文件和单字节快速边界）；
格式参数记录、识别与损坏拒绝。CPU 可控状态模型禁止调用完整 forward。
真实 GPU 正确性通过所选样本长度的独立进程编解码验证。旧测试保留，需要相关回归时按需运行。
