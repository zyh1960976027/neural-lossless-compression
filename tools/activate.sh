#!/usr/bin/env bash
# Optional compatibility shortcut; conda activate mamba3 is sufficient.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source tools/activate.sh" >&2
    exit 1
fi

_mamba3_conda_root="${MAMBA3_CONDA_ROOT:-$HOME/anaconda3}"
if [[ ! -f "$_mamba3_conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Conda not found at $_mamba3_conda_root; set MAMBA3_CONDA_ROOT." >&2
    unset _mamba3_conda_root
    return 1
fi

source "$_mamba3_conda_root/etc/profile.d/conda.sh"
if ! conda activate mamba3; then
    unset _mamba3_conda_root
    return 1
fi
unset _mamba3_conda_root
