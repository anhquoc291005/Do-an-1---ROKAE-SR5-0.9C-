#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
SCAN_PYTHON="${MUJOCO_PYTHON:-python3}"
exec "$SCAN_PYTHON" scan_path.py --view --loop "$@"
