#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
MUJOCO_PYTHON="${MUJOCO_PYTHON:-python3}"
if [[ ! -f generated/sr5_scene.xml ]]; then "$MUJOCO_PYTHON" build_model.py; fi
exec "$MUJOCO_PYTHON" simulate.py --view "$@"
