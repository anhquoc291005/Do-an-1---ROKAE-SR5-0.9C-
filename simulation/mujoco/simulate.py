#!/usr/bin/env python3
"""Actual mj_step integration with torque control, gravity and contacts."""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'trajectory/v_path'
RESULTS=ROOT/'results'
KP=np.array([600.,600.,400.,120.,100.,70.])
KD=np.array([50.,50.,35.,10.,8.,6.])


class Simulation:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'generated/sr5_scene.xml'))
        self.data=mujoco.MjData(self.model)
        self.trajectory=np.loadtxt(SOURCE/'results/trajectory.csv',delimiter=',',skiprows=1)
        self.targets=np.loadtxt(SOURCE/'generated/targets.csv',delimiter=',',skiprows=1)
        self.velocity=np.gradient(self.trajectory[:,1:7],self.trajectory[:,0],axis=0)
        self.joints=[self.model.joint(f'joint{i}') for i in range(1,7)]
        self.qadr=np.array([int(j.qposadr[0]) for j in self.joints])
        self.dadr=np.array([int(j.dofadr[0]) for j in self.joints])
        self.tcp=int(self.model.site('tcp').id)
        self.torch=int(self.model.body('welding_tcp').id)
        self.probe=int(self.model.body('contact_probe').id)
        self.limits=self.model.actuator_ctrlrange[:,1]
        self.last_force=np.zeros(6)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        # Initial pose assignment only; subsequent q values come from integration.
        self.data.qpos[self.qadr]=self.trajectory[0,1:7]
        mujoco.mj_forward(self.model,self.data)

    def reference(self,t):
        q=np.array([np.interp(t,self.trajectory[:,0],self.trajectory[:,j]) for j in range(1,7)])
        v=np.array([np.interp(t,self.trajectory[:,0],self.velocity[:,j]) for j in range(6)])
        if t>=self.trajectory[-1,0]:v*=0
        p=np.array([np.interp(t,self.targets[:,0],self.targets[:,j]) for j in range(1,4)])
        return q,v,p

    def step(self,disturb=False,control=True):
        q,v,_=self.reference(self.data.time)
        # qfrc_bias contains gravity and Coriolis/centrifugal terms.
        tau=KP*(q-self.data.qpos[self.qadr])+KD*(v-self.data.qvel[self.dadr])+self.data.qfrc_bias[self.dadr]
        self.data.ctrl[:]=np.clip(tau,-self.limits,self.limits) if control else 0
        self.data.xfrc_applied[self.torch]=0
        if disturb:self.data.xfrc_applied[self.torch,0]=10.  # world +X, N
        mujoco.mj_step(self.model,self.data)
        # Refresh derived positions for the integrated state before measuring/rendering.
        mujoco.mj_forward(self.model,self.data)
        return tau

    def contact_stats(self):
        robot_contacts=[];probe_contacts=0;max_force=0.
        for i,c in enumerate(self.data.contact):
            names=[self.model.geom(int(g)).name for g in (c.geom1,c.geom2)]
            if 'probe_cube' in names:probe_contacts+=1
            if any(name and name.startswith(('xMateSR5_','welding_tcp_')) for name in names):
                robot_contacts.append(names)
            if 'probe_cube' in names:
                mujoco.mj_contactForce(self.model,self.data,i,self.last_force)
                max_force=max(max_force,float(np.linalg.norm(self.last_force[:3])))
        return robot_contacts,probe_contacts,max_force


def validate(seconds=None,disturbance=False):
    RESULTS.mkdir(exist_ok=True)
    sim=Simulation();m,d=sim.model,sim.data
    # FK conversion validation at all supplied states, outside the physics run.
    max_fk=0.;max_j=0.
    source_fk=np.loadtxt(SOURCE/'results/fk.csv',delimiter=',',skiprows=1)
    source_j=np.loadtxt(SOURCE/'results/jacobian.csv',delimiter=',',skiprows=1)
    for i in range(0,len(source_fk),20):
        d.qpos[sim.qadr]=sim.trajectory[i,1:7];mujoco.mj_forward(m,d)
        max_fk=max(max_fk,float(np.linalg.norm(d.site_xpos[sim.tcp]-source_fk[i,2:5])))
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jp,jr,sim.tcp)
        max_j=max(max_j,float(np.max(np.abs(np.vstack([jp[:,sim.dadr],jr[:,sim.dadr]])-source_j[i,2:38].reshape(6,6)))))
    sim.reset()
    stop=float(seconds if seconds is not None else sim.trajectory[-1,0]+1.)
    max_error=0.;sum_error2=0.;max_qerror=0.;max_torque=np.zeros(6);robot_contact_steps=0;contact_pairs=set()
    probe_contact_steps=0;max_probe_force=0.;saturated=0;steps=0;rows=[];max_angle=0.;bounds_steps=0
    start=time.monotonic()
    while d.time<stop:
        kick=disturbance and 10<=d.time<10.5
        tau=sim.step(disturb=kick)
        q,_,p=sim.reference(d.time)
        error=float(np.linalg.norm(d.site_xpos[sim.tcp]-p))
        max_error=max(max_error,error);sum_error2+=error**2
        max_qerror=max(max_qerror,float(np.max(np.abs(d.qpos[sim.qadr]-q))))
        actual_R=d.site_xmat[sim.tcp].reshape(3,3)
        expected_R=np.diag([1.,-1.,-1.])
        max_angle=max(max_angle,float(np.arccos(np.clip((np.trace(expected_R.T@actual_R)-1)/2,-1,1))))
        qranges=np.array([j.range for j in sim.joints])
        bounds_steps+=int(np.any(d.qpos[sim.qadr]<qranges[:,0]-1e-6) or np.any(d.qpos[sim.qadr]>qranges[:,1]+1e-6))
        max_torque=np.maximum(max_torque,np.abs(d.ctrl));saturated+=int(np.any(np.abs(tau)>sim.limits))
        contacts,pc,pf=sim.contact_stats();robot_contact_steps+=bool(contacts);probe_contact_steps+=bool(pc);max_probe_force=max(max_probe_force,pf)
        contact_pairs.update(tuple(c) for c in contacts)
        if steps%10==0:rows.append([d.time,*q,*d.qpos[sim.qadr],*d.qvel[sim.dadr],*d.ctrl,*d.site_xpos[sim.tcp],error,pc])
        steps+=1
        if not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qvel))>100:raise RuntimeError('Unstable physics')
    final_probe_z=float(d.xpos[sim.probe,2])
    # With all six torques disabled, gravity must move the arm from this pose.
    probe_sim=Simulation();q0=probe_sim.data.qpos[probe_sim.qadr].copy()
    for _ in range(100):probe_sim.step(control=False)
    passive_motion=float(np.max(np.abs(probe_sim.data.qpos[probe_sim.qadr]-q0)))
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number}
    mode='disturbance' if disturbance else 'tracking'
    report={
        'mode':mode,'mujoco_version':mujoco.__version__,'simulated_s':float(d.time),'wall_s':time.monotonic()-start,'physics_steps':steps,
        'timestep_s':float(m.opt.timestep),'gravity_m_s2':m.opt.gravity.tolist(),
        'fk_conversion_max_error_m':max_fk,'jacobian_conversion_max_error':max_j,
        'max_tcp_tracking_error_mm':max_error*1000,'rms_tcp_tracking_error_mm':(sum_error2/steps)**.5*1000,
        'max_joint_tracking_error_deg':float(np.rad2deg(max_qerror)),
        'max_orientation_tracking_error_deg':float(np.rad2deg(max_angle)),
        'joint_limit_violation_steps':bounds_steps,'final_tcp_tracking_error_mm':error*1000,
        'peak_actuator_torque_Nm':max_torque.tolist(),'actuator_limits_Nm':sim.limits.tolist(),'saturated_steps':saturated,
        'robot_contact_steps':robot_contact_steps,'robot_contact_pairs':[list(c) for c in sorted(contact_pairs)],
        'falling_cube_contact_steps':probe_contact_steps,'falling_cube_peak_contact_force_N':max_probe_force,
        'falling_cube_final_center_z_m':final_probe_z,'expected_rest_center_z_m':.31,
        'unpowered_arm_max_joint_change_rad_after_0_2s':passive_motion,'warnings':warnings,
        'controller':'Torque motors: PD tracking plus model bias compensation, clipped to URDF effort limits',
        'disturbance':'10 N at torch along world X, t=10..10.5 s' if disturbance else None,
        'limitation':'URDF inertias unverified; assumed tool/damping/friction; convex robot contacts; plate groove visual only'}
    report['passed']=bool(max_fk<1e-8 and max_j<1e-8 and not warnings and not robot_contact_steps and not bounds_steps and passive_motion>1e-3
                          and probe_contact_steps>0 and abs(final_probe_z-.31)<.002
                          and (disturbance or max_error<.001) and not saturated)
    (RESULTS/f'{mode}_summary.json').write_text(json.dumps(report,indent=2))
    header=['time_s']+[f'{prefix}{i}' for prefix in ['qref_rad_','q_rad_','qvel_rad_s_','torque_Nm_'] for i in range(1,7)]+['tcp_x_m','tcp_y_m','tcp_z_m','tcp_error_m','probe_contacts']
    with (RESULTS/f'{mode}.csv').open('w') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit('Physics validation did not pass; inspect report')


def view(rate=1.):
    import mujoco.viewer
    sim=Simulation()
    paused=False;kick_until=0.
    def key(code):
        nonlocal paused,kick_until
        if code==32:paused=not paused
        if code in (ord('P'),ord('p')):kick_until=sim.data.time+.5
        if code in (ord('R'),ord('r')):sim.reset();kick_until=0.
    with mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key) as viewer:
        viewer.cam.lookat[:]=[.30,0,.38];viewer.cam.distance=1.65;viewer.cam.azimuth=135;viewer.cam.elevation=-28
        viewer.opt.geomgroup[3]=0
        frame=0
        print('MuJoCo SR5 physics | SPACE: pause | P: 10 N torch push | R: reset | close window: exit',flush=True)
        while viewer.is_running():
            started=time.monotonic()
            if not paused:
                for _ in range(max(1,round(rate/(60*sim.model.opt.timestep)))):
                    sim.step(disturb=sim.data.time<kick_until)
                    if sim.data.time>sim.trajectory[-1,0]+3:sim.reset();kick_until=0.
            viewer.sync();frame+=1
            if frame==60:
                RESULTS.mkdir(exist_ok=True)
                (RESULTS/'viewer_status.json').write_text(json.dumps({'viewer_open':True,'sim_time_s':sim.data.time,'physics':'mj_step with torque motors','rate':rate},indent=2))
            time.sleep(max(0,1/60-(time.monotonic()-started)))


def render():
    import struct
    import zlib
    sim=Simulation()
    # A settled physical state under gravity, then render without assigning qpos.
    while sim.data.time<1.:sim.step()
    renderer=mujoco.Renderer(sim.model,height=800,width=1280)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.30,0,.38]
    camera.distance=1.55;camera.azimuth=135;camera.elevation=-30
    option=mujoco.MjvOption();option.geomgroup[3]=0
    renderer.update_scene(sim.data,camera=camera,scene_option=option)
    RESULTS.mkdir(exist_ok=True)
    rgb=renderer.render()
    height,width,_=rgb.shape
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    scanlines=b''.join(b'\x00'+row.tobytes() for row in rgb)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(scanlines))+chunk(b'IEND',b'')
    (RESULTS/'mujoco_scene.png').write_bytes(png)
    renderer.close();print(RESULTS/'mujoco_scene.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--view',action='store_true');p.add_argument('--rate',type=float,default=1.)
    p.add_argument('--seconds',type=float);p.add_argument('--disturbance',action='store_true');p.add_argument('--render',action='store_true');args=p.parse_args()
    if args.rate<=0:raise SystemExit('--rate must be positive')
    if args.render:render()
    elif args.view:view(args.rate)
    else:validate(args.seconds,args.disturbance)
