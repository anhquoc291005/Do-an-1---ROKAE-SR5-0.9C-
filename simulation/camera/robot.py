from pathlib import Path
import json
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
KP=np.array([600.,600.,400.,120.,100.,70.])
KD=np.array([50.,50.,35.,10.,8.,6.])


def angle_error(Rd,R):
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,(Rd@R.T).reshape(-1))
    if quat[0]<0:quat=-quat
    v=np.zeros(3);mujoco.mju_quat2Vel(v,quat,1.)
    return v


def look_at(eye,target):
    z=target-eye;z/=np.linalg.norm(z)
    x=np.cross(z,np.array([0.,1.,0.]));x/=np.linalg.norm(x)
    return np.column_stack([x,np.cross(z,x),z])


class Robot:
    def __init__(self,part):
        self.part=part
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'generated'/f'{part}_scene.xml'))
        self.data=mujoco.MjData(self.model)
        self.sid=int(self.model.site('camera_optical').id)
        self.cid=int(self.model.camera('scan_camera').id)
        self.limits=self.model.actuator_ctrlrange[:,1]
        self.ranges=self.model.jnt_range.copy()
        self.profile=json.loads((ROOT/'generated/camera_profile.json').read_text())
        self.max_tau=np.zeros(6);self.saturated=0;self.contacts=0;self.max_v=0.;self.max_omega=0.

    def forward(self,q):
        self.data.qpos[:]=q;self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)
        return self.data.site_xpos[self.sid].copy(),self.data.site_xmat[self.sid].reshape(3,3).copy()

    def jacobian(self):
        jp=np.zeros((3,6));jr=np.zeros((3,6))
        mujoco.mj_jacSite(self.model,self.data,jp,jr,self.sid)
        return jp,jr

    def collision_pairs(self):
        result=[]
        for c in self.data.contact:
            names=[self.model.geom(int(g)).name for g in (c.geom1,c.geom2)]
            # Static/static scene contacts are filtered by MuJoCo; all remaining
            # contacts involve this robot/camera and must be rejected for scanning.
            if c.dist<0:result.append(names)
        return result

    def ik(self,p,R,seed):
        self.last_failure='iterations'
        q=np.clip(np.array(seed,dtype=float),self.ranges[:,0]+.001,self.ranges[:,1]-.001)
        for iteration in range(250):
            actual,rotation=self.forward(q)
            e=np.r_[p-actual,.3*angle_error(R,rotation)]
            if np.linalg.norm(e[:3])<2e-5 and np.linalg.norm(e[3:])<2e-5:
                if not self.collision_pairs():return q
                self.last_failure=str(self.collision_pairs())
                return None
            jp,jr=self.jacobian();J=np.vstack([jp,.3*jr])
            dq=J.T@np.linalg.solve(J@J.T+1e-6*np.eye(6),e)
            dq*=min(1.,.12/max(np.max(np.abs(dq)),1e-12))
            improved=False
            for scale in [1.,.5,.25,.1]:
                candidate=np.clip(q+scale*dq,self.ranges[:,0]+.001,self.ranges[:,1]-.001)
                pp,RR=self.forward(candidate)
                ee=np.r_[p-pp,.3*angle_error(R,RR)]
                if np.linalg.norm(ee)<np.linalg.norm(e):q=candidate;improved=True;break
            if not improved:
                self.last_failure=f'no improvement: poserr={np.linalg.norm(e[:3]):.4f}, roterr={np.linalg.norm(e[3:])/.3:.4f}'
                return None
        return None

    def reset(self,q):
        mujoco.mj_resetData(self.model,self.data);self.data.qpos[:]=q
        mujoco.mj_forward(self.model,self.data)

    def step(self,q,v):
        tau=KP*(q-self.data.qpos)+KD*(v-self.data.qvel)+self.data.qfrc_bias
        self.data.ctrl[:]=np.clip(tau,-self.limits,self.limits)
        self.max_tau=np.maximum(self.max_tau,np.abs(self.data.ctrl));self.saturated+=int(np.any(np.abs(tau)>self.limits))
        mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
        self.contacts+=bool(self.collision_pairs())
        jp,jr=self.jacobian();speed=float(np.linalg.norm(jp@self.data.qvel));omega=float(np.linalg.norm(jr@self.data.qvel))
        self.max_v=max(self.max_v,speed);self.max_omega=max(self.max_omega,omega)
        return speed,omega


def plan(part):
    robot=Robot(part);rng=np.random.default_rng(11)
    target=np.array([.4,0.,.30])
    offsets=[(0,0,.40),(-.22,0,.34),(-.16,-.16,.34),(0,-.22,.34),
             (.12,0,.34),(.10,.14,.32),(0,.22,.34),(-.16,.16,.34)]
    qprev=np.array([2.95,-.2,1.6,-np.pi,-1.34,2.95])
    accepted=[];rejected=[]
    for index,offset in enumerate(offsets):
        eye=target+np.array(offset);R=look_at(eye,target)
        seeds=[qprev,[0,.5,1.5,0,1,0]]+[rng.uniform(np.maximum(robot.ranges[:,0],-3.1),np.minimum(robot.ranges[:,1],3.1)) for _ in range(24)]
        found=None
        for seed in seeds:
            q=robot.ik(eye,R,seed)
            if q is None:continue
            # Revolute ±2pi equivalents: choose the closest legal representation.
            if accepted:
                for j in range(6):
                    options=[q[j]+k*2*np.pi for k in [-1,0,1] if robot.ranges[j,0]<q[j]+k*2*np.pi<robot.ranges[j,1]]
                    q[j]=min(options,key=lambda a:abs(a-qprev[j]))
                collision=False;max_path_derivative=0.
                for alpha in np.linspace(0,1,151):
                    robot.forward((1-alpha)*qprev+alpha*q)
                    if robot.collision_pairs():collision=True;break
                    jp,_=robot.jacobian();max_path_derivative=max(max_path_derivative,float(np.linalg.norm(jp@(q-qprev))))
                if collision:continue
                duration=max(3.,1.875*np.max(np.abs(q-qprev))/.25,1.875*max_path_derivative/.065)
            else:duration=0.
            found={'id':len(accepted),'candidate_id':index,'eye_target_world_m':eye.tolist(),
                   'look_target_world_m':target.tolist(),'R_world_optical':R.tolist(),
                   'q_rad':q.tolist(),'transition_duration_s':float(duration)}
            break
        if found is None:rejected.append({'candidate_id':index,'reason':'No sampled collision-free IK/connection found; not proof of infeasibility','last_failure':robot.last_failure})
        else:accepted.append(found);qprev=np.array(found['q_rad'])
        print(f'{part} candidate {index}: '+('accepted' if found else str(rejected[-1])),flush=True)
    report={'part':part,'planner':'MuJoCo geometric Jacobian DLS IK, multi-seed; sampled joint-space connections',
            'viewpoints':accepted,'rejected_candidates':rejected,'connection_samples':151,
            'transition_peak_optical_speed_limit_m_s':.08,
            'notes':'Predefined view template, not optimized coverage or adaptive next-best-view; static phôi pose known'}
    path=ROOT/'generated'/f'{part}_scan_plan.json';path.write_text(json.dumps(report,indent=2))
    if len(accepted)<4:raise RuntimeError(f'Only {len(accepted)} valid scan views: {rejected}')
    print(f'{part}: {len(accepted)} reachable views, {len(rejected)} rejected candidates; {path}',flush=True)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['tee','v_plate','all'],default='all');a=p.parse_args()
    for part in ['tee','v_plate'] if a.part=='all' else [a.part]:plan(part)
