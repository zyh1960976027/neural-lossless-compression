# BPE 概率量化与基线测速

当前主线只维护 BPE 4096 的 GPU 整数量化。模型预测概率后，GPU 按原有双精度累计顺序生成 24 bit 整数频率，每个子词至少获得一个频率单位；编码仅传输实际符号的区间，解码传输完整累计表。整数熵编码接口复用 Constriction 0.4.2 的范围编码核心，由 `tools/build_integer_range.py` 构建。

```bash
python tools/build_integer_range.py
python tools/benchmark_baseline.py --input 10MB样本路径 --output-dir experiments/test/新测速目录
```

测速默认批大小 1024、分词线程 24、PyTorch CPU 线程 1；每轮编码和解码均运行于独立进程，先按完整批和尾批形状预热，再记录操作时间，默认三轮取吞吐中位数。每次均核对恢复字节，重复编码要求相同码流摘要。输出目录必须不存在，目录中保存任务、日志、结果及 `summary.json`。

通过 `--reference 对照目录/summary.json` 比较相同输入、权重、分词器及批大小，码流不一致或任一中位吞吐回退超过 5% 时返回失败。正式比较还须使用相同 GPU 和运行环境。源码职责见[目录说明](../目录说明.md)，整理前后实测见[主线整理报告](../实验报告/主线整理/实验报告.md)。FP32/FP16、其他词表及旧对照脚本仅在[历史版本](历史索引.md)中运行。
