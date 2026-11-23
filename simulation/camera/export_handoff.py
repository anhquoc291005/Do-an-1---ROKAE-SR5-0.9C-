#!/usr/bin/env python3
"""Camera pose / robot joint / trigger interface for the image-processing team."""
import csv
import json
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent


def export(part):
    out=ROOT/'results'/part;s=json.loads((out/'summary.json').read_text())
    if not s['passed']:raise RuntimeError(f'{part}: refusing to export a failed trajectory')
    fields=['view_id','trigger_time_s','hold_end_s','camera_x_target_m','camera_y_target_m','camera_z_target_m',
            'camera_qw_target','camera_qx_target','camera_qy_target','camera_qz_target']+[f'joint{i}_target_rad' for i in range(1,7)]+['measured_settle_s']
    with (out/'camera_waypoints.csv').open('w') as f:
        w=csv.writer(f);w.writerow(fields)
        for v in s['viewpoints']:
            quat=np.zeros(4);mujoco.mju_mat2Quat(quat,np.array(v['R_world_optical']).ravel())
            w.writerow([v['id'],v['trigger_time_s'],v['hold_end_time_s'],*v['eye_target_world_m'],*quat,*v['q_rad'],v['settle_s']])
    print(out/'camera_waypoints.csv')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['tee','v_plate','all'],default='all');a=p.parse_args()
    for part in ['tee','v_plate'] if a.part=='all' else [a.part]:export(part)
