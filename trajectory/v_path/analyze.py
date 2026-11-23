#!/usr/bin/python3
"""Independent URDF transform-product FK/Jacobian and numerical diagnostics."""
from pathlib import Path
import json
import xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


import sys
sys.path.insert(0, str(ROOT.parents[1] / "forward_kinematics"))
from fk import Robot as UrdfRobot, origin

class Robot(UrdfRobot):
    def __init__(self):
        super().__init__(ROOT / "generated/sr5_torch.urdf")


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    robot = Robot()
    result = ROOT / 'results'
    raw = np.loadtxt(result / 'trajectory.csv', delimiter=',', skiprows=1)
    fk = np.loadtxt(result / 'fk.csv', delimiter=',', skiprows=1)
    jac = np.loadtxt(result / 'jacobian.csv', delimiter=',', skiprows=1)
    target = np.loadtxt(ROOT / 'generated/targets.csv', delimiter=',', skiprows=1)
    t, q = raw[:, 0], raw[:, 1:7]
    fd_position, fd_angle, j_error = [], [], []
    for i in range(len(q)):
        T,J,_ = robot.fk_jacobian(q[i])
        fd_position.append(np.linalg.norm(T[:3,3]-fk[i,2:5]))
        fd_angle.append((Rotation.from_matrix(T[:3,:3]).inv()*Rotation.from_quat(fk[i,5:9])).magnitude())
        j_error.append(np.max(np.abs(J-jac[i,2:38].reshape(6,6))))
    v = np.gradient(q,t,axis=0,edge_order=2)
    a = np.gradient(v,t,axis=0,edge_order=2)
    J = jac[:,2:38].reshape(-1,6,6)
    speed = np.linalg.norm(np.einsum('nij,nj->ni',J[:,:3,:],v),axis=1)
    max_v = np.max(np.abs(v),axis=0)
    max_a = np.max(np.abs(a),axis=0)
    metrics = {
        'independent_fk_max_position_error_m': float(max(fd_position)),
        'independent_fk_max_orientation_error_rad': float(max(fd_angle)),
        'independent_geometric_jacobian_max_error': float(max(j_error)),
        'peak_joint_velocity_rad_s': max_v.tolist(),
        'peak_joint_acceleration_fd_rad_s2': max_a.tolist(),
        'peak_tcp_speed_from_Jqdot_mm_s': float(max(speed)*1000),
        'acceleration_note': 'Finite differences of sampled IK; not a validated continuous controller command',
        'demo_acceleration_limit_rad_s2': 0.5,
        'max_velocity_limit_ratio': float(max(max_v)/np.pi),
        'max_demo_acceleration_ratio': float(max(max_a)/0.5),
    }
    (result/'independent_checks.json').write_text(json.dumps(metrics,indent=2))
    if max(fd_position)>1e-10 or max(fd_angle)>1e-10 or max(j_error)>1e-10:
        raise RuntimeError('Independent URDF check disagrees with MoveIt')
    if max(max_v)>np.pi or max(max_a)>0.5 or max(speed)>0.0051:
        raise RuntimeError('Velocity or demonstration acceleration limit exceeded')
    header = 'time_s,'+','.join(f'joint{i}_{suffix}' for suffix in ('rad','rad_s','rad_s2') for i in range(1,7))
    np.savetxt(result/'joint_diagnostics.csv',np.c_[t,q,v,a],delimiter=',',header=header,comments='')
    fig,axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for j in range(6):
        axes[0,0].plot(t,np.rad2deg(q[:,j]),label=f'J{j+1}')
        axes[0,1].plot(t,np.rad2deg(v[:,j]),label=f'J{j+1}')
    axes[0,0].set(ylabel='Joint angle (deg)',title='SR5 joint trajectory')
    axes[0,0].legend(ncol=3)
    axes[0,1].set(ylabel='Joint speed (deg/s)',title='Numerical joint velocities')
    axes[1,0].plot(t,fk[:,9]*1000,color='#1a7a91')
    axes[1,0].set(ylabel='Position residual (mm)',title='FK(IK(target)) residual')
    axes[1,1].plot(t,speed*1000,label='TCP speed (mm/s)')
    axes[1,1].plot(t,jac[:,-1],label='Minimum scaled singular value')
    axes[1,1].legend()
    axes[1,1].set(title='Speed and Jacobian conditioning')
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Time (s)')
    fig.savefig(result/'kinematics.png',dpi=160)
    (result/'jacobian_start.txt').write_text('Spatial geometric Jacobian at first trajectory point; rows [vx vy vz wx wy wz], world frame.\n'+np.array2string(J[0],precision=9))
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
