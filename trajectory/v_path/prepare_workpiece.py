#!/usr/bin/env python3
"""A planar V path on a CAD plate with a matching shallow tracking groove.

This locally generated teaching fixture is not a seam label from the upstream
dataset. V describes the path in top view, not the weld groove cross-section.
"""
from pathlib import Path
import csv
import json
import numpy as np
import cadquery as cq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
out = ROOT / 'generated'
out.mkdir(exist_ok=True)
work_origin = np.array([0.50, 0., 0.30])
local_mm = np.array([[-70., 55., 0.], [0., -55., 0.], [70., 55., 0.]])
plate = cq.Workplane('XY').box(260, 220, 8, centered=(True,True,False)).translate((0,0,-8))
# A shallow 3 mm wide channel shows where the torch is tracing the V.
for a,b in zip(local_mm[:-1], local_mm[1:]):
    center = (a+b)/2
    delta = b-a
    length_mm = float(np.linalg.norm(delta))
    angle = float(np.rad2deg(np.arctan2(delta[1],delta[0])))
    channel = (cq.Workplane('XY').box(length_mm+3,3,3,centered=(True,True,False))
               .translate((0,0,-2.5)).rotate((0,0,0),(0,0,1),angle)
               .translate(tuple(center)))
    plate = plate.cut(channel)
cq.exporters.export(plate,str(out/'v_path_plate.step'))
cq.exporters.export(plate,str(out/'workpiece_mm.stl'),tolerance=0.1,angularTolerance=0.1)
seam = local_mm*.001+work_origin
axis = np.array([0.,0.,-1.])
rotation = np.diag([1.,-1.,-1.])
quat = Rotation.from_matrix(rotation).as_quat()
tcp = seam-.005*axis
lengths = np.linalg.norm(np.diff(seam,axis=0),axis=1)
durations = 1.875*lengths/.005
rows=[]
start=0.
corner_pause=.6
for segment,(a,b,duration) in enumerate(zip(tcp[:-1],tcp[1:],durations)):
    samples=int(np.ceil(duration/.05))+1
    for k,t in enumerate(np.linspace(0,float(duration),samples)):
        if segment and k==0:continue
        u=t/duration;s=10*u**3-15*u**4+6*u**5
        rows.append([float(start+t),*(a+s*(b-a)),*quat])
    start+=float(duration)
    if segment==0:
        for t in np.linspace(.05,corner_pause,12):
            rows.append([float(start+t),*b,*quat])
        start+=corner_pause
with (out/'targets.csv').open('w') as f:
    writer=csv.writer(f)
    writer.writerow(['time_s','x_m','y_m','z_m','qx','qy','qz','qw'])
    writer.writerows(rows)
scene={
    'part_id':'v_path_plate', 'path_name':'V-shaped welding path',
    'path_shape':'V', 'fixture_source':'Locally generated CadQuery teaching fixture; not upstream dataset ground truth',
    'source_units':'mm','world_units':'m','mesh_scale':.001,
    'work_origin_m':work_origin.tolist(),
    'T_world_workpiece':[[1,0,0,.5],[0,1,0,0],[0,0,1,.3],[0,0,0,1]],
    'seam_endpoints_world_m':seam.tolist(),
    'tcp_endpoints_world_m':tcp.tolist(),
    'torch_axis_world':axis.tolist(),'quaternion_xyzw':quat.tolist(),
    'standoff_m':.005,'torch_flange_offset_m':.18,
    'seam_length_m':float(lengths.sum()),'segment_lengths_m':lengths.tolist(),
    'duration_s':start,'peak_tcp_speed_m_s':.005,
    'corner_time_s':float(durations[0]),'corner_pause_s':corner_pause,
    'table':{'center':[.5,0.,.267],'size':[.34,.30,.05]},
    'bbox_mm':[-130,130,-110,110,-8,0],
    'notes':'Two planar straight legs, fixed downward torch, quintic progress per leg, rest for 0.6 s at vertex. Kinematic path tracking only.'}
(out/'scene.json').write_text(json.dumps(scene,indent=2))
(out/'path_waypoints.csv').write_text('point,x_m,y_m,z_m\n'+'\n'.join(f'{name},{p[0]},{p[1]},{p[2]}' for name,p in zip('ABC',seam))+'\n')
print(json.dumps(scene,indent=2))
