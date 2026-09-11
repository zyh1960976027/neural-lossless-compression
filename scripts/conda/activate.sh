#!/usr/bin/env bash
# Sourced automatically by Conda; installed by install_conda_hooks.sh.

_mamba3_activate() {
    local project_root name backup
    project_root=$(cd -P -- "$CONDA_PREFIX/../.." && pwd) || return

    # Keep unset variables distinct from variables set to an empty string.
    if [[ -z ${_MAMBA3_SAVED_VARIABLES:-} ]]; then
        export _MAMBA3_SAVED_VARIABLES="PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONNOUSERSITE BYTE_EXPERIMENT_PYTHON CUDA_HOME TORCH_CUDA_ARCH_LIST TORCH_EXTENSIONS_DIR PIP_CACHE_DIR TRITON_CACHE_DIR LIBRARY_PATH CUDA_VISIBLE_DEVICES"
        for name in $_MAMBA3_SAVED_VARIABLES; do
            backup="_MAMBA3_SAVED_$name"
            if [[ -v $name ]]; then
                export "$backup=${!name}"
            else
                unset "$backup"
            fi
        done
    fi

    unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
    export PYTHONNOUSERSITE=1
    export BYTE_EXPERIMENT_PYTHON="$CONDA_PREFIX/bin/python"
    export CUDA_HOME=/usr/local/cuda-12.1
    export TORCH_CUDA_ARCH_LIST=8.6
    export TORCH_EXTENSIONS_DIR="$project_root/.torch_extensions"
    export PIP_CACHE_DIR="$project_root/.cache/pip"
    export TRITON_CACHE_DIR="$project_root/.cache/triton"
    # The stub is for linking only; do not put it in LD_LIBRARY_PATH.
    case ":${LIBRARY_PATH:-}:" in
        *":$CUDA_HOME/targets/x86_64-linux/lib/stubs:"*) ;;
        *) export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" ;;
    esac
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
}

_mamba3_activate
unset -f _mamba3_activate
