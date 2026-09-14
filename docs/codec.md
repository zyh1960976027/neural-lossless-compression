# 无损编解码

本模块复用现有 Mamba-3 权重，通过 Constriction 0.4.2 的
`RangeEncoder` / `RangeDecoder` 生成和读取真实 `.m3rc` 压缩文件。
模型、训练、数据集和原有 BPB 评估代码不变。

## 使用

在项目根目录激活已有环境，安装新增依赖：

```bash
conda activate mamba3
python -m pip install -r requirements-codec.txt
```

一次完成编码、解码、逐字节比较和指标输出：

```bash
python codec.py roundtrip \
  --input /path/to/input.bin \
  --compressed /path/to/input.m3rc \
  --decoded /path/to/decoded.bin \
  --checkpoint checkpoints/best-3M.pt \
  --results /path/to/results.json
```

也可在不同进程分别执行：

```bash
python codec.py encode \
  --input /path/to/input.bin --output /path/to/input.m3rc \
  --checkpoint checkpoints/best-3M.pt

python codec.py decode \
  --input /path/to/input.m3rc --output /path/to/decoded.bin \
  --checkpoint checkpoints/best-3M.pt

cmp /path/to/input.bin /path/to/decoded.bin
```

解码不需要原文件或编码时保存的概率表，但必须提供同一份 checkpoint。
`--device` 默认 `cuda:0`；真实 Mamba-3 要求支持 BF16 的 CUDA GPU。
`--checkpoint` 的默认路径与原有评估相同，仍为 `checkpoints/best.pt`。
当前本机权重名是 `best-3M.pt`，所以示例显式指定它。
2 层和 4 层权重均由原有加载函数读取。

所有输出路径的父目录须已存在；已有文件不会被覆盖。
编码或解码失败时删除本次生成的不完整输出。
结果 JSON 输出到 stdout；简要指标输出到 stderr；`--results` 可选。
默认输出紧凑的 **v2** 格式。编码和 roundtrip 可用 `--format v1` 重现旧格式，
或 `--format v2-framed` 只精简文件头、保留原有分帧，用于分别比较两项优化。
解码根据文件 magic 自动识别，不需要指定格式。详细环境信息保留在结果 JSON 中；
用 `--results` 保存实验日志，解码本身无需该日志。

## 分块与概率

沿用 `ByteEvaluationDataset(sequence_length=2048)` 的语义：

1. 整个文件的 byte 0 按 `1/256` 均匀概率编码，只计一次 8-bit 理论成本。
2. 第一个预测块编码 byte 1..2048，以 byte 0 为输入起点。
3. 下一个预测块编码 byte 2049..4096，以已经解出的 byte 2048 为输入起点。
4. 每个预测块只使用起点 byte 和块内已知历史，重置所有其他上下文。
   尾块仅编码真实存在的 byte，零填充不写入码流或恢复文件。

因此这里的 2048 是每块**预测目标数**，相邻块共享一个上下文 byte。
模型上下文每个预测块重置；默认 v2 的 Range Coder 在整个文件中保持连续，
首字节也编码进同一个流，只在文件结束时收尾一次。
v1 和 v2-framed 每个预测块重新创建 Range Coder。
文件长度为 N 时，有 `max(N-1, 0)` 个模型预测目标，以及
`ceil(max(N-1, 0)/2048)` 个预测块。空文件和单字节文件均支持。

两端都逐 byte 构造 `[1, 2048]` 输入：已知历史在前，未知部分补零；
调用现有模型后取最后一个已知位置的 256 类 logits。
编码端也只向模型提供这一历史，不把尚未编码的真实 byte 填进模型。
每一步重新运行完整前向，不使用缓存状态或专用 step 内核。
这保证两端走相同路径，但速度较慢，适合先验证正确性。

推理使用 eval、inference mode、BF16 autocast、FP32 模型参数，关闭 TF32，
开启 PyTorch 确定性选项。256 类 logits 转到 CPU，以 float64 计算
log-softmax、NLL 和概率。两端显式使用 `Categorical(probabilities, perfect=False)`。
Constriction 将概率转换为可逆的定点分布，并为全部 256 个符号保留非零质量，
因此浮点概率下溢至零的 byte 仍可编码。NaN/Inf logits 直接报错。
参考 [Categorical 官方说明](https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html)
和 [Range Coding 官方接口](https://bamler-lab.github.io/constriction/apidoc/python/stream/queue.html)。

## 文件格式 v2（默认）

所有二进制整数为小端序。固定开销为 **106 bytes**，与文件大小和块数无关：

| 部分 | bytes | 内容 |
| --- | ---: | --- |
| magic | 8 | `M3RC\x02\r\n\x1a`，包含格式版本 |
| 概率配置编号 | 1 | 当前为 1，对应现有 PROFILE、2048 目标分块和原有 CDF |
| Range 流模式 | 1 | 1 为全文件共享流；0 为 v2-framed 对比分帧模式 |
| 原始长度 | 8 | uint64，控制解码总数及尾块长度 |
| 模型指纹 | 32 | 完整 checkpoint 文件的 SHA256 二进制摘要 |
| 环境指纹 | 16 | 环境 JSON 规范序列化后 SHA256 的前 16 bytes |
| 文件头 CRC32 | 4 | 校验前面的 66 bytes，以上合计 70 bytes |
| 编码主体 | 可变 | v2 是单个 uint32 Range 流，没有逐块帧头 |
| 主体 CRC32 | 4 | 校验整个编码主体 |
| 原文件 SHA256 | 32 | 检查最终恢复结果，尾部合计 36 bytes |

环境指纹由 `json.dumps(runtime, sort_keys=True, separators=(",", ":"))` 的 UTF-8
字节计算，详细 runtime 字段与 v1 一致，包括库版本和 GPU 型号。
这样仍能在解码前拒绝不匹配的运行环境，详细版本文本则放入结果日志。
解码不需要把环境指纹反查成原始 JSON。

主体长度由实际文件大小减去 106 得到，原始长度决定符号数，不需要 EOS 或逐块长度。
空文件主体为零 bytes；非空主体必须按 4 bytes 对齐。
头部与主体 CRC32 在模型解码前验证，最后验证恢复文件的 SHA256。
该格式未加入随机访问索引；模型继续使用原来的块边界和历史，跨块仅延续熵编码器状态。

v2-framed 使用完全相同的 70-byte 头和 36-byte 尾，主体保留 v1 的全部帧头、
逐帧 CRC32 和逐帧 Range payload。因此它还有 `8 × 帧数` bytes 额外开销，
仅用于格式对比或需要分帧的场景。主体 CRC32 此时也覆盖帧头。

当前 Python 接口会把共享 Range 流保存在内存中，内存占用随压缩数据长度增长；
模型历史仍限制在 2048 个输入位置。这次优化没有实现大文件码流的流式 I/O。

## 文件格式 v1（兼容）

全部二进制整数使用小端序。布局按以下顺序保存：

| 部分 | 内容 |
| --- | --- |
| 固定头 | 8 字节 magic `M3RC\r\n\x1a\n`，4 字节 JSON 长度 |
| JSON 头 | 格式版本、原始字节数、2048 块长、概率计算协议、checkpoint SHA256、运行环境版本 |
| 首字节帧 | 非空文件才有，使用均匀模型的独立 Range Coder |
| 预测块帧 | 每帧 4 字节 payload 长度、4 字节 CRC32，随后是小端 uint32 码流 |
| 文件尾 | 原文件 SHA256，32 字节二进制摘要 |

首字节帧和预测块帧使用相同帧格式。原始长度决定解码符号数和尾块长度，
不使用 EOS。头部和帧长度有上限检查；损坏头部、版本/权重/运行环境不匹配、
截断、CRC32/SHA256 校验失败及多余尾部数据均报错。
`roundtrip` 在解码自身的 SHA256 校验以外，还直接逐段比较原文件和解码文件。

不在压缩文件中内嵌模型。checkpoint 指纹针对完整 checkpoint 文件，
即使模型相同，重新打包 checkpoint 也会产生不同指纹。
两种版本均要求相同 GPU 型号、PyTorch/CUDA/Mamba/NumPy/Triton/Constriction 版本。
版本匹配和确定性开关不能保证所有第三方 GPU 内核在所有机器上逐位一致；
最终 SHA256 校验用于阻止不同概率导致的错误结果被当成成功输出。
暂不承诺跨硬件或跨软件版本解码。

## 指标

| 字段 | 计算方式 |
| --- | --- |
| `theoretical_nll_bpb` | `(文件首字节 ln(256) + 编码时实际使用的逐 byte NLL) / (原始字节数 × ln(2))` |
| `actual_bpb` | `整个 .m3rc 文件字节数 × 8 / 原始字节数`，包含头、帧、收尾和校验 |
| `payload_bpb` | 只计 Range Coder 输出的 uint32 数据，包括每帧收尾/对齐开销 |
| `encode_kb_per_second` | `原始字节数 / 1000 / encode_seconds` |
| `decode_kb_per_second` | `恢复字节数 / 1000 / decode_seconds` |
| `files_identical` | `roundtrip` 对两个文件的直接字节比较结果 |

计时包含读取、模型推理、概率处理、熵编解码、校验和写入，GPU 在开始和结束时同步。
不含模型加载及 checkpoint 指纹计算，不包含 roundtrip 的额外文件比较。
未单独预热，首次编译可能使编码速度低于解码速度。1 KB = 1000 字节。
写入耗时包括用户态文件关闭，不强制磁盘 fsync。

理论 NLL 在 Constriction 概率量化前计算；定点量化和文件格式开销使其不同于实际 BPB。
它与现有评估具有相同分块/目标计数口径，但原有评估一次输入完整块、使用 FP32
交叉熵；本模块逐步补零并使用 float64 NLL，因此 BF16 内核和浮点归约可能带来数值差异。
不能要求两个 BPB 逐位相等。需要比较时请对相同文件显式设置评估 batch size 为 1。
空文件理论 BPB 为 0，实际 BPB 和 payload BPB 为 `null`（分母为零），速度为 0。
小文件的格式开销占比较高，不保证压缩文件比原文件小。

## 测试

```bash
python -m unittest discover -s tests -p test_codec.py -v
```

CPU 测试覆盖空/单字节、2048 附近边界、多块及尾块、0..255、历史重置、
非均匀及极小概率、旧评估 NLL 口径、重复编码、单独解码、文件损坏和覆盖保护。
同时检查三个格式的 NLL/历史一致、v1 与 v2-framed 的帧字节完全相同、
v2 固定开销为 106 bytes，以及配置编号、流模式和校验损坏的处理。
这些测试使用可控概率模型；真实 Mamba-3 验证使用上述 CLI 和实际 checkpoint。
