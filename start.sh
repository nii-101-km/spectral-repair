#!/usr/bin/env bash
set -euo pipefail
project_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_dir/var/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$project_dir/.venv/bin/python" "$project_dir/launcher.py" "$@"
