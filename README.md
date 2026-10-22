# neural-lossless-compression（神经网络无损压缩）

本项目使用神经网络预测概率进行无损压缩。当前主线固定为 **BPE 4096（最大编号 4095）＋两层 Mamba-3（6,158,796 参数）＋GPU 整数量化＋Constriction 范围编码**。BPE 将原始字节表示为可逆的子词序列；模型预测下一个子词的概率，在 GPU 上量化为整数频率，Constriction 根据这些频率生成压缩码流。解码使用相同权重逐步复现预测并恢复原始字节。

## 运行准备

需要 Linux、已配置的 Mamba-3/PyTorch 环境、支持 BF16 的 CUDA 显卡，以及 Rust/Cargo 1.75 或更新版本。在项目根目录执行：

```bash
conda activate mamba3
python -m pip install -r requirements-codec.txt
python tools/build_integer_range.py
```

配套的分词器和推理权重默认位于 `checkpoints/bpe4096/tokenizer.model`、`checkpoints/bpe4096/inference.pt`，本机已关联到选定的实验文件。数据、权重、完整实验材料、运行环境和本机编译的共享库不随 Git 分发，协作者需另行取得或生成。环境配置见[运行环境](docs/使用指南/运行环境.md)。

## 压缩与解压

在已激活的环境中执行：

```bash
python run.py encode --input datasets/raw/enwik9 --output experiments/test/enwik9.m3rc
python run.py decode --input experiments/test/enwik9.m3rc --output experiments/test/enwik9.decoded
```

一次完成压缩、解压和原文逐字节校验：

```bash
python run.py roundtrip --input datasets/raw/enwik9 \
  --compressed experiments/test/enwik9.m3rc --decoded experiments/test/enwik9.decoded
```

已有输出不会覆盖，父目录自动创建。编码默认使用 `cuda:0`、批大小 1024、分词线程 24、PyTorch CPU 线程 1；可以用 `--checkpoint`、`--tokenizer`、`--device`、`--batch-size` 调整配套文件和运行参数。解码从码流读取分词器与批大小，要求相同权重和兼容环境。运行后打印 JSON，`--results 路径` 可另存指标；`encode` 报告包含分词器和容器开销的完整文件 BPB。

## 训练与评估

使用 enwik9 前 100 MB 训练、随后 10 MB 验证；省略 `--source` 时使用同一个本机默认路径：

```bash
python run.py train --source datasets/raw/enwik9 --run-dir experiments/train/bpe4096_new
```

使用自行准备的完整训练、验证文件：

```bash
python run.py train --train-file 数据路径 --validation-file 验证数据路径 --run-dir 实验目录
```

两种数据方式互斥。程序依次准备 BPE、验证分词可逆性、生成两份 token 缓存、完成训练预检并训练。分词器只从训练部分学习；小语料无法构建完整 4096 词表时会报错，可以用 `--tokenizer checkpoints/bpe4096/tokenizer.model` 复用已有分词器。默认每块 2048 个 token、有效批大小 72、加载进程数 0；`--batch-size`、`--micro-batch-size` 和 `--device` 可调整训练配置。`--max-epochs 1` 用于短训练检查，默认按验证集早停收敛。

重复使用同一实验目录会在数据、分词器和配置一致时自动续训，变更则拒绝复用。目录内保存 `checkpoints/best.pt`、`checkpoints/latest.pt`、`inference.pt`、训练曲线、配置及摘要；新权重不会替换默认基线。评估概率 BPB：

```bash
python run.py evaluate --input 数据路径 \
  --checkpoint 实验目录/inference.pt --tokenizer 实验目录/tokenizer.model
```

评估按独立块计算，屏蔽尾部填充，并计入每块首 token 的均匀成本。该指标用于判断模型概率质量，完整压缩文件 BPB 由 `encode` 报告。详细产物和续训规则见[训练与评估](docs/使用指南/训练与评估.md)。

## 结构与历史

根目录唯一 Python 操作入口是 `run.py`；`python -m neural_compression` 调用同一实现，五个子命令均支持 `--help`。核心代码集中在 `neural_compression/`，开发工具位于 `tools/`，职责和调用方式见[目录说明](docs/目录说明.md)。

长期维护五个分支：`main`、`experiment/unigram`、`baseline/byte`、`experiment/mtp`、`experiment/model-scale`。旧字节格式、FP32/FP16 子词格式和多方案实验脚本通过历史分支、标签保存；运行历史命令须检出对应版本。历史报告、精简数据、保留材料及清理记录见[历史索引](docs/使用指南/历史索引.md)，协作约定见[版本管理与协作](docs/使用指南/版本管理与协作.md)。

## 验证

```bash
python -m unittest discover -s tests -v
MAMBA_CODEC_GPU_TEST=1 python -m unittest discover -s tests -v
```

普通检查覆盖整数编码、可逆分词、数据校验、训练保存恢复和推理导出；第二条额外运行真实 GPU、默认权重边界及预取检查。[主线整理验证报告](docs/实验报告/主线整理/实验报告.md)记录 10 MB 三次独立进程对照、码流一致性和本次清理结果。当前算法的原始实验结果见[GPU 整数量化报告](docs/实验报告/BPEGPU整数量化/实验报告.md)。
