#!/usr/bin/env bash
# Source this file from Bash to activate the project's Mamba-3 runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source scripts/activate.sh" >&2
    exit 1
fi

_mamba3_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_mamba3_conda_root="${MAMBA3_CONDA_ROOT:-$HOME/anaconda3}"
if [[ ! -f "$_mamba3_conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Conda not found at $_mamba3_conda_root; set MAMBA3_CONDA_ROOT." >&2
    unset _mamba3_project_root _mamba3_conda_root
    return 1
fi

# Prevent another Python environment from supplying Torch or Mamba imports.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
source "$_mamba3_conda_root/etc/profile.d/conda.sh"
if ! conda activate mamba3; then
    unset _mamba3_project_root _mamba3_conda_root
    return 1
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export BYTE_EXPERIMENT_PYTHON="$CONDA_PREFIX/bin/python"
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.6
export TORCH_EXTENSIONS_DIR="$_mamba3_project_root/.torch_extensions"
export PIP_CACHE_DIR="$_mamba3_project_root/.cache/pip"
export TRITON_CACHE_DIR="$_mamba3_project_root/.cache/triton"
# Triton needs the link-time libcuda stub; runtime still uses the host driver.
case ":${LIBRARY_PATH:-}:" in
    *":$CUDA_HOME/targets/x86_64-linux/lib/stubs:"*) ;;
    *) export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" ;;
esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset _mamba3_project_root _mamba3_conda_root
