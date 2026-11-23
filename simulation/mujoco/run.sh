#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
MUJOCO_PYTHON="${MUJOCO_PYTHON:-python3}"
"$MUJOCO_PYTHON" build_model.py
"$MUJOCO_PYTHON" simulate.py "$@"
