# 运行环境

本机的 `mamba3` 环境已安装 Conda 激活脚本。在 Bash 终端中，只需：

```bash
conda activate mamba3
```

随后在项目根目录运行训练或评估。将测试文件路径换成真实文件：

```bash
python evaluate.py \
  --test-file /path/to/test.bin \
  --name debug
```

默认使用 `checkpoints/best.pt`、`cuda:0` 和序列长度 `2048`。
结果自动保存到 `experiments/test/<时间戳>_debug/results.json` 和 `summary.csv`。
完整参数及训练产物目录见 [评估说明](evaluation.md)。

不再需要手动执行 `source scripts/activate.sh`。该脚本保留为兼容入口。
如果终端之前已经激活了 `mamba3`，请重新执行一次 `conda activate mamba3`。
此处假设终端已经初始化 Conda，能够正常使用 `conda activate`。

## 自动配置的内容

激活时设置 CUDA 12.1 编译路径、GPU 架构、项目缓存目录和 Python 依赖隔离，
并为链接器添加 CUDA stub 库目录。已有的 `CUDA_VISIBLE_DEVICES` 非空设置
会保留，否则默认使用设备 0。退出环境时运行 `conda deactivate`，这些变量
恢复到激活前的值。

环境实际位于 `.conda-envs/mamba3/`；本机通过 Conda 环境目录里的符号链接
将它登记为 `mamba3`。配置路径按照该项目布局计算，支持通过环境名称或实际
环境路径激活。

## 重建环境后的一次性安装

环境目录不提交 Git。重新创建本项目的 `.conda-envs/mamba3/` 环境后，或修改
`scripts/conda/` 中的配置后，在项目根目录执行一次：

```bash
bash scripts/install_conda_hooks.sh
conda activate mamba3
```

安装脚本将受版本管理的配置复制到环境的 `etc/conda/activate.d/` 和
`etc/conda/deactivate.d/`，之后由 Conda 自动加载。当前配置面向本机 Linux/Bash
和 CUDA 12.1；其他机器需按实际 CUDA 安装路径和 GPU 架构调整。

机制参考：[Conda 官方文档](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#saving-environment-variables)。
