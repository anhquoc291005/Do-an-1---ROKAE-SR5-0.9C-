#!/usr/bin/python3
"""Static scientific plots of scan motion and nominal weld visibility."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main(part):
    out=ROOT/'results'/part;s=json.loads((out/'summary.json').read_text())
    with (out/'joint_trajectory.csv').open() as f:rows=list(csv.DictReader(f))
    t=np.array([float(r['time_s']) for r in rows]);phase=np.array([r['phase'] for r in rows])
    p=np.array([[float(r[k]) for k in ['camera_x_m','camera_y_m','camera_z_m']] for r in rows])
    speed=np.linalg.norm(np.gradient(p,t,axis=0),axis=1)*1000
    fig=plt.figure(figsize=(13,8),layout='constrained')
    ax=fig.add_subplot(221,projection='3d');ax.plot(*p.T,color='#2783b4',lw=1.5)
    for v in s['viewpoints']:
        eye=np.array(v['eye_target_world_m']);aim=np.array(v['look_target_world_m'])
        ax.scatter(*eye,color='#d67d10');ax.text(*eye,str(v['id']),fontsize=11)
        ax.plot(*np.array([eye,aim]).T,color='#d67d10',alpha=.3,lw=.7)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title=f'SR5 optical-center path: {part}')
    ax.view_init(28,-125)
    ax=fig.add_subplot(222)
    for j in range(1,7):ax.plot(t,[np.rad2deg(float(r[f'q_actual_rad_{j}'])) for r in rows],label=f'J{j}')
    ax.set(xlabel='Simulation time (s)',ylabel='Joint position (deg)',title='Physical robot motion');ax.legend(ncol=3);ax.grid(alpha=.2)
    ax=fig.add_subplot(223);ax.plot(t,speed,color='#1b8464')
    for v in s['viewpoints']:ax.axvline(v['trigger_time_s'],color='#d67d10',alpha=.6,lw=.8)
    ax.set(xlabel='Simulation time (s)',ylabel='Optical-center speed (mm/s)',title='Move, settle, hold; vertical lines = triggers');ax.grid(alpha=.2)
    ax=fig.add_subplot(224)
    seams=sorted(s['coverage_by_seam_id'])
    for j,seam in enumerate(seams):
        for v in s['viewpoints']:
            if seam in v['visible_seam_ids']:ax.scatter(v['id'],j,s=140,marker='s',color='#2783b4')
    ax.set(xticks=range(s['views']),yticks=range(len(seams)),yticklabels=seams,xlabel='Camera view ID',ylabel='Known weld seam ID',title='Blue = at least some seam samples visible')
    ax.grid(alpha=.2)
    fig.suptitle(f'Camera trajectory only | {s["views"]} views | {s["duration_s"]:.1f} s | nominal sampled weld coverage {s["seam_sample_coverage"]:.0%}')
    fig.savefig(out/'trajectory_report.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,i in zip(axes,[1,4]):
        ax.imshow(plt.imread(out/f'camera_view_{i:02d}.png'));ax.axis('off');ax.set_title(f'Virtual camera view {i} | {part}')
    fig.suptitle('Geometric viewing check; not a prediction of real camera image quality')
    fig.savefig(out/'camera_views.png',dpi=150);plt.close(fig)
    print(out/'trajectory_report.png')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['tee','v_plate','all'],default='all');a=p.parse_args()
    for part in ['tee','v_plate'] if a.part=='all' else [a.part]:main(part)
