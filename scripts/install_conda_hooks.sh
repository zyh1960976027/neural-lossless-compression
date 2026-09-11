#!/usr/bin/env bash
# Install the Bash activation hooks into this project's existing Conda env.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
env_prefix="$project_root/.conda-envs/mamba3"
if [[ ! -d "$env_prefix/conda-meta" ]]; then
    echo "Missing Conda environment: $env_prefix" >&2
    exit 1
fi

for action in activate deactivate; do
    destination="$env_prefix/etc/conda/$action.d"
    mkdir -p -- "$destination"
    install -m 644 -- "$project_root/scripts/conda/$action.sh" \
        "$destination/mamba3-byte-compress.sh"
done
echo "Installed environment hooks. Run: conda activate mamba3"
