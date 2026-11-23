#!/usr/bin/env python3
"""SR5 only: gravity, torque-controlled pose hold, and interactive joint targets."""
import argparse
import json
import threading
import time
import xml.etree.ElementTree as E
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
HOME=np.deg2rad([0,30,-60,0,-90,0])
KP=np.array([600.,600.,400.,120.,100.,70.])
KD=np.array([50.,50.,35.,10.,8.,6.])


def build():
    tree=E.parse(ROOT/'generated/sr5_scene.xml');root=tree.getroot()
    root.set('model','SR5_robot_only')
    world=root.find('worldbody')
    for item in list(world):
        if item.tag!='light' and item.get('name')!='xMateSR5_base':world.remove(item)
    for parent in root.iter('body'):
        for item in list(parent):
            if item.tag=='body' and item.get('name')=='welding_tcp':parent.remove(item)
    contact=root.find('contact')
    for item in list(contact):
        if 'welding_tcp' in [item.get('body1'),item.get('body2')]:contact.remove(item)
    for item in list(root.find('asset')):
        if item.get('name') in ['checker','floor_mat','workpiece']:root.find('asset').remove(item)
    root.find('statistic').set('center','0.28 0 0.46')
    root.find('statistic').set('extent','1.3')
    E.SubElement(root,'keyframe')
    E.SubElement(root.find('keyframe'),'key',name='bent',qpos=' '.join(map(str,HOME)))
    E.indent(tree,space='  ')
    path=ROOT/'generated/sr5_robot_only.xml';tree.write(path,encoding='unicode')
    return path


class RobotOnly:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(build()))
        self.data=mujoco.MjData(self.model)
        self.qadr=np.array([self.model.joint(f'joint{i}').qposadr[0] for i in range(1,7)])
        self.dadr=np.array([self.model.joint(f'joint{i}').dofadr[0] for i in range(1,7)])
        self.aids=np.array([self.model.actuator(f'joint{i}_torque').id for i in range(1,7)])
        self.limits=self.model.actuator_ctrlrange[self.aids,1]
        self.ranges=np.array([self.model.joint(f'joint{i}').range for i in range(1,7)])
        self.target=HOME.copy();self.reference=HOME.copy()
        self.data.qpos[self.qadr]=HOME
        mujoco.mj_forward(self.model,self.data)
        self.max_tau=np.zeros(6);self.saturated=0;self.contacts=0

    def step(self,target):
        dt=self.model.opt.timestep
        # Low-pass command changes to keep interactive five-degree steps gradual.
        velocity=np.clip(2*(target-self.reference),-.20,.20)
        self.reference+=velocity*dt
        tau=KP*(self.reference-self.data.qpos[self.qadr])+KD*(velocity-self.data.qvel[self.dadr])+self.data.qfrc_bias[self.dadr]
        self.saturated+=int(np.any(np.abs(tau)>self.limits))
        self.data.ctrl[self.aids]=np.clip(tau,-self.limits,self.limits)
        self.max_tau=np.maximum(self.max_tau,np.abs(self.data.ctrl[self.aids]))
        mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
        self.contacts+=int(any(c.dist<0 for c in self.data.contact))
        if not np.all(np.isfinite(self.data.qpos)):raise RuntimeError('Nonfinite robot state')


def check():
    robot=RobotOnly()
    for _ in range(1000):robot.step(HOME)
    err=float(np.max(np.abs(robot.data.qpos[robot.qadr]-HOME)))
    report={'python_environment':'Python 3.12 project venv','mujoco_version':mujoco.__version__,
        'joints':robot.model.njnt,'motors':robot.model.nu,'initial_q_deg':np.rad2deg(HOME).tolist(),
        'simulated_seconds':float(robot.data.time),'maximum_final_joint_error_rad':err,
        'robot_contact_steps':robot.contacts,'saturated_steps':robot.saturated,
        'peak_torque_Nm':robot.max_tau.tolist(),
        'passed':bool(err<.001 and not robot.contacts and not robot.saturated and not any(w.number for w in robot.data.warning))}
    (ROOT/'results/robot_only_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit('Robot-only validation failed')


def view():
    import mujoco.viewer
    robot=RobotOnly();lock=threading.Lock();state={'selected':0,'target':HOME.copy(),'paused':False}
    def key(code):
        with lock:
            if ord('1')<=code<=ord('6'):
                state['selected']=code-ord('1');print(f'Selected joint {state["selected"]+1}',flush=True)
            elif code in [ord('+'),ord('='),ord('-')]:
                j=state['selected'];delta=np.deg2rad(-5 if code==ord('-') else 5)
                state['target'][j]=np.clip(state['target'][j]+delta,robot.ranges[j,0]+.01,robot.ranges[j,1]-.01)
                print('Target angles (deg):',np.round(np.rad2deg(state['target']),1),flush=True)
            elif code in [ord('R'),ord('r')]:state['target']=HOME.copy()
            elif code==32:state['paused']=not state['paused']
    with mujoco.viewer.launch_passive(robot.model,robot.data,key_callback=key) as viewer:
        viewer.cam.lookat[:]=[.27,.025,.45];viewer.cam.distance=1.85
        viewer.cam.azimuth=135;viewer.cam.elevation=-22
        viewer.opt.geomgroup[3]=0
        print('SR5 ONLY | Python 3.12 | Torque control + gravity | 1..6: select joint; +/-: 5 deg; R: bent pose; SPACE: pause',flush=True)
        wall=time.monotonic();start_sim=robot.data.time;last_sync=wall
        while viewer.is_running():
            with lock:target=state['target'].copy();paused=state['paused']
            if paused:
                viewer.sync();time.sleep(.016);wall=time.monotonic();start_sim=robot.data.time;continue
            robot.step(target)
            now=time.monotonic()
            if now-last_sync>=1/60:
                viewer.sync();last_sync=now
            sleep=wall+(robot.data.time-start_sim)-time.monotonic()
            if sleep>0:time.sleep(sleep)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check() if args.check else view()
