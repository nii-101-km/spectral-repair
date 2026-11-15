#!/usr/bin/env bash
set -euo pipefail
project_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_dir/var/matplotlib}"
exec "$project_dir/.venv/bin/python" "$project_dir/test.py" "$@"
