#!/usr/bin/env bash
# Restore the environment that existed before conda activate mamba3.

_mamba3_deactivate() {
    local name backup
    for name in ${_MAMBA3_SAVED_VARIABLES:-}; do
        backup="_MAMBA3_SAVED_$name"
        if [[ -v $backup ]]; then
            export "$name=${!backup}"
        else
            unset "$name"
        fi
        unset "$backup"
    done
    unset _MAMBA3_SAVED_VARIABLES
}

_mamba3_deactivate
unset -f _mamba3_deactivate
