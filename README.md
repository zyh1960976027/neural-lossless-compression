# neural-lossless-compression（神经网络无损压缩）

本项目研究基于神经网络预测概率的无损压缩。主线使用词表大小为 4096 的 BPE 子词分词器、一维 Mamba-3 序列模型，以及 Range Coding（范围编码，根据预测概率生成压缩码流的方法）。模型输出转换为子词概率后，在 GPU 上量化为整数频率与累计区间，再由 CPU 上的 Constriction 0.4.2 完成范围编码。解码使用相同权重和已经恢复的子词重现概率，最终还原原始字节。

项目名描述压缩任务，具体模型名称保留在实现和实验配置中。当前性能基准是已经完成无损验证的 GPU 整数量化版本，结果见[实验报告](docs/实验报告/BPEGPU整数量化/实验报告.md)。

## 方案分支

| 分支 | 维护内容 |
| --- | --- |
| `main` | BPE 4096、Mamba-3、GPU 概率整数量化与范围编码 |
| `experiment/unigram` | Unigram 4096 子词方案 |
| `baseline/byte` | 字节模型与优化后的 v5 编解码基线 |
| `experiment/mtp` | 多 token 预测实验 |
| `experiment/model-scale` | 模型深度与参数规模实验 |

整理前的分支节点通过 `archive/*` 标签保存，GPU 量化基准由 `baseline/bpe4096-gpu-int-20261021` 标签固定。分支来源及合作约定见[版本管理与协作](docs/使用指南/版本管理与协作.md)。

## 运行准备

当前实现需要 Linux、已配置的 Mamba-3/PyTorch 环境、支持 BF16 的 CUDA 显卡，以及 Rust/Cargo 1.75 或更新版本。所有命令均在项目根目录执行：

```bash
conda activate mamba3
python -m pip install -r requirements-codec.txt
python scripts/build_integer_range.py
```

`requirements-codec.txt` 补充范围编码与分词依赖；最后一条命令构建 CPU 整数范围编码接口。当前机器的 Conda 环境仍名为 `mamba3`，环境位置与配置见[运行环境](docs/使用指南/运行环境.md)。

准备配套的 BPE 4096 分词器和推理权重，再运行下方示例。数据集、权重、完整实验产物和本机编译的共享库由本地管理，不随 Git 仓库分发；协作者需要另行取得这些文件。

## BPE 编码与解码

把示例中的权重、分词器和输入输出路径替换为实际位置：

```python
from neural_compression.token_codec import (
    encode, decode, load_token_predictor, load_token_decoder,
)
from neural_compression.tokenization import LosslessTokenizer
from neural_compression.codec import verify_files

checkpoint = "/path/to/bpe4096/inference.pt"
tokenizer = LosslessTokenizer("/path/to/bpe4096/tokenizer.model", threads=24)
predictor = load_token_predictor(checkpoint, probability_transfer="gpu_int")
encode("input.bin", "compressed.m3rc", predictor, tokenizer)

# 解码也可在独立进程中执行，分词器和编解码设置从压缩文件读取。
decoder = load_token_decoder(checkpoint, "compressed.m3rc")
decode("compressed.m3rc", "decoded.bin", decoder)
verify_files("input.bin", "decoded.bin")
```

每块包含最多 2048 个子词，默认同时处理 1024 块。编码端只下载实际子词对应的整数区间；解码端下载完整整数累计表，由 CPU 查找并恢复子词。压缩文件内嵌分词器，记录批大小、量化方式、权重和运行环境指纹，并通过长度、CRC 和原文 SHA256 校验恢复结果。

主线调用须显式选择 `gpu_int`；底层接口为兼容已有调用保留 FP32 默认值。解码要求相同权重和兼容运行环境，输出父目录须存在，已有文件不会覆盖。模式、格式和复现实验说明见[BPE 概率传输](docs/使用指南/BPE概率传输.md)。

## 训练与其他入口

BPE 和 Unigram 的数据准备、训练及评测使用 `scripts/benchmark_sentencepiece.py`，执行顺序及恢复要求见[子词与字节压缩对照实验](docs/实验方案/子词与字节压缩对照实验.md)。

根目录的 `codec.py`、`train.py` 和 `evaluate.py` 提供字节基线的编解码、训练和概率评估。字节权重与子词权重须分别使用；具体命令见[无损编解码](docs/使用指南/无损编解码.md)和[训练与评估](docs/使用指南/训练与评估.md)。

## 目录与文档

```text
neural-lossless-compression/
├── README.md                 中文项目入口
├── AGENTS.md                 项目协作规范
├── neural_compression/       分词、预测、训练与范围编解码实现
├── native/integer_range/     CPU 整数范围编码接口及锁定的 Rust 依赖
├── scripts/                  实验、性能分析、构建及环境脚本
├── tests/                    自动化检查
├── docs/                     使用指南、实验方案、报告、精简数据与图片
├── codec.py                  字节基线编解码入口
├── train.py                  字节基线训练入口
├── evaluate.py               字节基线概率评估入口
├── requirements-codec.txt    编解码附加依赖
├── datasets/                 本地数据集，Git 忽略
├── checkpoints/              本地模型权重，Git 忽略
└── experiments/              本地运行产物，Git 忽略大型文件
```

完整文件职责和文档索引见[目录说明](docs/目录说明.md)。当前主线的实现包为 `neural_compression`；其他方案分支沿用各自快照中的源码布局，切换后以该分支说明为准。实验结论与历史代码指纹保存在原报告和结果文件中，新实验使用独立运行目录。

## 检查

```bash
python -m unittest discover -s tests -v
```

普通检查使用受控的小模型验证训练、量化、格式、边界和损坏检测；真实 GPU 检查按相应指南和测试的环境要求显式开启。GPU 整数量化实验中的 18 次完整往返均通过内部校验和外部逐字节比较，详细数据见[GPU 整数量化实验报告](docs/实验报告/BPEGPU整数量化/实验报告.md)。
