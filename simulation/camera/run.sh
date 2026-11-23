#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
SCAN_PYTHON="${MUJOCO_PYTHON:-python3}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$SCAN_PYTHON" build_scene.py
"$SCAN_PYTHON" robot.py --part all
"$SCAN_PYTHON" scan_path.py "$@"
