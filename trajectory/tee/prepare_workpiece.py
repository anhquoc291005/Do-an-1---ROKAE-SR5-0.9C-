#!/usr/bin/env python3
"""Generate a real upstream coupon, export mesh and timed Cartesian targets.

Run with ../../.venv-cad/bin/python. All upstream CAD coordinates are millimetres.
The executable experiment intentionally accepts one straight, planar fillet seam.
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import cadquery as cq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'external/weldment-dataset'))
from weldment.generator.batch import generate_one

out = ROOT / 'generated'
out.mkdir(exist_ok=True)
result = generate_one((0, 'coupon_tee', 42, str(out)))
if not result['ok']:
    raise RuntimeError(result)
stem = 'p0000_coupon_tee'
labels = json.loads((out / f'{stem}_welds.json').read_text())
seams = [s for s in labels['candidate_seams'] if s['is_candidate_weld'] and s['curve_type'] == 'Line' and s['length'] > 50 and len(s['supporting_normals']) == 2]
if not seams:
    raise RuntimeError('No suitable straight fillet seam in generated coupon')
# Front-facing side of the upright, towards a robot at world origin.
seam = min(seams, key=lambda s: s['mid_point'][0])
p = np.array(seam['polyline'], dtype=float) * 0.001
if len(p) != 2:
    raise RuntimeError('This demo expects the two endpoints of a line')
if p[0, 1] > p[-1, 1]:
    p = p[::-1]
work_origin = np.array([0.50, 0., 0.30])
p += work_origin
tangent = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
normals = np.array(seam['supporting_normals'], dtype=float)
normals /= np.linalg.norm(normals, axis=1)[:, None]
outward = normals.sum(axis=0)
outward /= np.linalg.norm(outward)
axis = -outward  # local torch +Z points from the flange towards the seam
axis -= tangent * np.dot(axis, tangent)
axis /= np.linalg.norm(axis)
rot = np.column_stack((tangent, np.cross(axis, tangent), axis))
quat = Rotation.from_matrix(rot).as_quat()
standoff = 0.005
tcp = p - standoff * axis
length = float(np.linalg.norm(p[1] - p[0]))
# Quintic path progress: rest at endpoints, peak speed 5 mm/s.
duration = 1.875 * length / 0.005
n = int(np.ceil(duration / 0.05)) + 1
times = np.linspace(0., duration, n)
with (out / 'targets.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw'])
    for time in times:
        u = time / duration
        s = 10*u**3 - 15*u**4 + 6*u**5
        w.writerow([time, *(tcp[0] + s*(tcp[1]-tcp[0])), *quat])
part = cq.importers.importStep(str(out / f'{stem}.step'))
cq.exporters.export(part, str(out / 'workpiece_mm.stl'), tolerance=0.1, angularTolerance=0.1)
bb = part.val().BoundingBox()
scene = {
    'part_id': stem, 'seed': 42, 'generation': result, 'selected_edge_id': seam['edge_id'],
    'joint_type': seam['joint_type'], 'source_units': 'mm', 'world_units': 'm',
    'T_world_workpiece': [[1,0,0,0.5], [0,1,0,0], [0,0,1,0.3], [0,0,0,1]],
    'mesh_scale': 0.001, 'work_origin_m': work_origin.tolist(),
    'seam_endpoints_world_m': p.tolist(), 'tcp_endpoints_world_m': tcp.tolist(),
    'torch_axis_world': axis.tolist(), 'quaternion_xyzw': quat.tolist(),
    'supporting_normals': normals.tolist(), 'standoff_m': standoff,
    'torch_flange_offset_m': 0.18, 'seam_length_m': length,
    'duration_s': duration, 'peak_tcp_speed_m_s': 0.005,
    'bbox_mm': [bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax],
    'table': {'center': [0.5, 0, 0.3+bb.zmin*0.001-0.025],
              'size': [bb.xlen*0.001+0.08, bb.ylen*0.001+0.08, 0.05]},
    'notes': 'One straight seam; bisector work angle, zero travel angle; illustrative torch and process speed.'}
(out / 'scene.json').write_text(json.dumps(scene, indent=2))
print(json.dumps(scene, indent=2))
