#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=90
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
/usr/bin/python3 prepare_robot.py
if [[ ! -f generated/scene.json || "${1:-}" == "--regenerate" ]]; then
  "${CAD_PYTHON:-../../.venv-cad/bin/python}" prepare_workpiece.py > generated/workpiece_generation.log 2>&1
fi
/usr/bin/python3 - <<'PY'
import json,yaml
from pathlib import Path
p=Path('generated');s=json.loads((p/'scene.json').read_text())
params={'table_center':[float(x) for x in s['table']['center']], 'table_size':[float(x) for x in s['table']['size']],
        'workpiece_origin':[float(x) for x in s['work_origin_m']]}
(p/'scene_params.yaml').write_text(yaml.safe_dump({'/**':{'ros__parameters':params}}))
Path('results').mkdir(exist_ok=True)
Path('results/summary.json').write_text('{"success":false,"status":"run in progress or failed"}\n')
PY
cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_IGNORE_PREFIX_PATH="${CONDA_PREFIX:-}" > results/build.log 2>&1
cmake --build build -j4 >> results/build.log 2>&1
./build/sr5_demo --ros-args --params-file generated/params.yaml --params-file generated/scene_params.yaml 2>&1 | tee results/run.log
/usr/bin/python3 analyze.py | tee results/analysis.log
