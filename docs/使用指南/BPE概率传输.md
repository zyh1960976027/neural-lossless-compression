# BPE 概率传输

BPE 编解码支持 `fp32`、`fp16` 和 `gpu_int` 三种传输方式。模型仍生成 FP32 概率；`fp16` 在 GPU 上将概率转换为 16 位浮点数，传回 CPU 后恢复为 FP32，再由原 Constriction 0.4.2 完成概率量化与区间编解码。`gpu_int` 在 GPU 上生成整数频率与累计区间，再交给 CPU 上同版本的 Constriction 区间编解码核心。各模式编码和解码执行相同规则，原始文件须逐字节无损恢复。

主分支采用 BPE 4096 和 `gpu_int`，根目录 `run.py` 命令入口默认选择该模式；底层 Python API 为兼容已有调用仍保留 `fp32` 默认值。FP16 的传输量减半，但 GPU 转换和 CPU 恢复也有成本，对照结果见[FP16 实验报告](../实验报告/BPE传输优化/实验报告.md)。根目录 `codec.py` 是字节基线入口。

## 命令行使用

激活 `mamba3` 环境后，在项目根目录执行：

```bash
python run.py encode --input datasets/raw/enwik9 --output experiments/test/enwik9.m3rc
python run.py decode --input experiments/test/enwik9.m3rc --output experiments/test/enwik9.decoded
```

默认权重为 `checkpoints/bpe4096/inference.pt`，分词器为 `checkpoints/bpe4096/tokenizer.model`。本机已关联到现有 BPE 4096 实验文件；其他机器需准备对应文件，或使用 `--checkpoint`、`--tokenizer` 指定路径。输出父目录自动创建，已有文件不会覆盖；`--results` 可另存 JSON 指标。编码默认批大小 1024、分词线程 24、PyTorch CPU 线程 1，可通过相应参数调整。解码从压缩文件读取批大小和概率处理方式，不需要提供外部分词器。

需要对照 FP32 或 FP16 时，在编码命令追加 `--probability-transfer fp32` 或 `--probability-transfer fp16`。`python run.py roundtrip --input <原文> --compressed <压缩文件> --decoded <恢复文件>` 可完成往返与逐字节比较，完整参数见 `python run.py encode --help`。

## GPU 整数量化

`gpu_int` 首次使用前需构建本项目的 CPU 整数接口。Linux 环境安装 Rust/Cargo 工具链后，在项目根目录运行：

```bash
python scripts/build_integer_range.py
```

构建使用 `native/integer_range/Cargo.lock` 固定的依赖，其中 Constriction 为 0.4.2；生成的共享库不纳入版本管理。仅使用 FP32、FP16 或字节路径时无需 Rust 工具链。下方示例按主线设置为 `gpu_int`，解码自动读取文件头。该模式使用 CUDA Graph 执行模型推理、整数量化及目标区间提取。

量化总频率为 2²⁴，每个 token 至少分配 1，其余频率根据归一化累计概率分配。FP32 模型概率通过 FP64 前缀和计算累计概率，取整得到累计频率；相邻累计频率之差为符号频率。编码与解码采用相同运算顺序，GPU 同时检查原始概率合法性和整数累计表严格递增。模型计算精度不变，新增 FP64 只用于量化。

编码端已知实际 token，只下载其区间起点和频率，每行 8 字节，满批 1024 行为 8 KiB。解码端还不知道 token，下载完整的 32 位整数累计表，由 CPU 二分查找并更新区间解码状态；每行 V+1 个整数，满批约为 16／32／64 MiB。熵编码状态和码流读写均留在 CPU。CPU 扩展通过批量接口复用 Constriction 编码器，不在 Python 中逐 token 回调。

`gpu_int` 使用子词容器 v3，记录传输模式、24 位频率精度及量化规则 `cdf-f64-leaky24-v1`，与旧概率量化结果不保证码流一致。错误模式或量化参数会被拒绝。整数模式的理论负对数指标按实际频率除以 2²⁴ 计算，浮点零概率计数不再采集，记为 null。

## 编码与解码

```python
from neural_compression.token_codec import (
    load_token_predictor, load_token_decoder, encode, decode,
)
from neural_compression.tokenization import LosslessTokenizer

checkpoint = "experiments/tokenizers/enwik9_sp4096_20261019/evaluation/bpe/inference.pt"
tokenizer_path = "experiments/tokenizers/enwik9_sp4096_20261019/bpe/tokenizer.model"
predictor = load_token_predictor(checkpoint, probability_transfer="gpu_int")
tokenizer = LosslessTokenizer(tokenizer_path, threads=24)
encoded = encode("input.bin", "compressed.m3rc", predictor, tokenizer)

# 分开进程执行时，解码器从压缩文件自动读取批大小和传输精度。
decoder = load_token_decoder(checkpoint, "compressed.m3rc")
decoded = decode("compressed.m3rc", "decoded.bin", decoder)
```

输入、输出路径必须不同，输出父目录须存在，已有输出不会覆盖。解码内部验证长度、CRC 和原文 SHA256；还可使用 `neural_compression.codec.verify_files` 将恢复文件与原文逐字节比较。同一预测器创建后不能修改传输精度，切换模式须重新创建。

FP32 沿用子词容器 v1。FP16 使用 v2 和独立标识，并将 `probability_transfer=fp16` 写入文件头。使用错误精度的预测器会在解码前被拒绝；旧程序不能读取新 FP16 文件。`read_token_settings` 可单独读取批大小和传输配置。

结果中的 `probability_transfer_bytes` 是本次编解码提交的 GPU→CPU 概率表载荷字节数，包含固定批形状中仍传输的无效行，不包含模型加载、预热、输入 token 上传和概率检查标志，也不是 PCIe 总线实际流量计数器。编码的两套缓冲区不会让同一概率表重复计数。

## 复现实验

在项目 `mamba3` 环境、可访问 CUDA 的终端中执行，输出目录应使用新名称：

```bash
python scripts/benchmark_probability_transfer.py benchmark \
  --output experiments/tokenizers/probability_transfer_reproduce
```

默认对 4096、8192、16384 词表各执行 FP32、FP16 三轮独立进程编解码。使用相同 10,000,000 字节原文，批大小 1024，分词线程 24，PyTorch CPU 线程 1；模型加载和预热单列，正式编码仍从原文实际分词。可通过 `--vocab-sizes`、`--probability-transfer`、`--repeats`、`--device` 缩小测试或指定设备。

对照 GPU 整数量化时执行：

```bash
python scripts/benchmark_probability_transfer.py benchmark \
  --probability-transfer fp32 gpu_int \
  --output experiments/tokenizers/gpu_integer_reproduce
```

该对照同样运行三个词表、各模式三轮，并检查编码端实际下载量为 FP32 的 2/V，解码端为 (V+1)/V。`probability_transfer_bytes` 在整数模式下统计整数区间或累计表的载荷，不含编码目标 token 上传及检查标志。共享库二进制、Rust 源码与依赖锁文件也纳入实验代码指纹。

GPU 整数模式的实际 BPB、速度、重复耗时范围及无损验证见[GPU 整数量化实验报告](../实验报告/BPEGPU整数量化/实验报告.md)。

每次保留任务配置、日志、结果 JSON、压缩文件和恢复文件，结束后生成 `summary.json` 与 `summary.csv`。相同配置及代码指纹的已完成任务可以续用；配置或代码改变必须使用新目录。汇总会复核保留文件的摘要，并检查相同模式重复编码一致、FP16 概率载荷恰为 FP32 一半。

完整验收要求三个词表全部无损、BPB 增幅不超过 1%，且编码和解码各三次 FP16 耗时均低于对应 FP32 三次耗时，才具备切换默认模式的依据。单轮试跑不会满足完整验收。
