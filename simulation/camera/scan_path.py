#!/usr/bin/env python3
"""Plan/validate camera motion and trigger poses only; no point-cloud pipeline."""
import argparse
import csv
import json
import struct
import time
import zlib
from pathlib import Path
import numpy as np
import mujoco
from robot import ROOT, Robot, angle_error


def png(path,rgb):
    h,w,_=rgb.shape
    def chunk(k,b):return struct.pack('!I',len(b))+k+b+struct.pack('!I',zlib.crc32(k+b)&0xffffffff)
    b=b''.join(b'\x00'+row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b))+chunk(b'IEND',b''))


def welds(part):
    report=json.loads((ROOT/'generated'/f'{part}_scene_report.json').read_text())
    source=Path(report['source_scene']).parent
    if part=='tee':
        labels=json.loads((source/'p0000_coupon_tee_welds.json').read_text())
        paths=[{'id':str(s['edge_id']),'polyline':np.array(s['polyline'])*.001+report['work_origin_m']} for s in labels['candidate_seams'] if s['is_candidate_weld']]
    else:
        scene=json.loads((source/'scene.json').read_text())
        p=np.array(scene['seam_endpoints_world_m'])+report['shift_from_welding_demo_m']
        # The earlier welding target was on the nominal plate surface. Visibility
        # must test an actual surface: the existing V channel floor is 2.5 mm lower.
        p[:,2]-=.0025
        paths=[{'id':'AB','polyline':p[:2]},{'id':'BC','polyline':p[1:]}]
    return paths


def samples(paths):
    result=[];ids=[]
    for path in paths:
        for a,b in zip(path['polyline'][:-1],path['polyline'][1:]):
            # Exclude the exact endpoints, whose visibility is ambiguous on sharp boundaries.
            n=max(5,int(np.ceil(np.linalg.norm(b-a)/.002)))
            for u in np.linspace(.01,.99,n):result.append((1-u)*a+u*b);ids.append(path['id'])
    return np.array(result),np.array(ids)


def visible(robot,points):
    p=robot.data.site_xpos[robot.sid];R=robot.data.site_xmat[robot.sid].reshape(3,3)
    local=(points-p)@R;z=local[:,2]
    profile=robot.profile['simulation'];w,h=profile['resolution']
    u=profile['fx']*local[:,0]/np.maximum(z,1e-9)+w/2
    v=profile['fy']*local[:,1]/np.maximum(z,1e-9)+h/2
    mask=(z>=.30)&(z<=1.1)&(u>=8)&(u<w-8)&(v>=8)&(v<h-8)
    hits=np.zeros(len(points),dtype=bool)
    geomgroup=np.array([1,1,1,0,1,1],dtype=np.uint8)
    for i in np.flatnonzero(mask):
        delta=points[i]-p;distance=np.linalg.norm(delta);direction=delta/distance
        gid=np.array([-1],dtype=np.int32)
        hit=mujoco.mj_ray(robot.model,robot.data,p,direction,geomgroup,1,int(robot.model.body('industrial_camera').id),gid)
        hits[i]=bool(hit>=0 and abs(hit-distance)<.002 and int(gid[0])==int(robot.model.geom('workpiece_visual').id))
    return hits,local


def run(part,viewer=False,rate=1.,loop=False):
    robot=Robot(part);m,d=robot.model,robot.data
    plan=json.loads((ROOT/'generated'/f'{part}_scan_plan.json').read_text())
    out=ROOT/'results'/(part+'_preview' if viewer else part);out.mkdir(parents=True,exist_ok=True)
    # Remove stale success before beginning a new validation run.
    (out/'summary.json').write_text(json.dumps({'passed':False,'status':'running or interrupted'}))
    refs=plan['viewpoints'];robot.reset(np.array(refs[0]['q_rad']))
    points,ids=samples(welds(part));observed=np.zeros(len(points),dtype=bool)
    perview=[];logs=[];triggers=[];viewer_handle=None;renderer=None
    if viewer:
        from mujoco import viewer as viewer_api
        viewer_handle=viewer_api.launch_passive(m,d)
        viewer_handle.cam.lookat[:]=[.33,0,.5];viewer_handle.cam.distance=1.8
        viewer_handle.cam.azimuth=135;viewer_handle.cam.elevation=-28;viewer_handle.opt.geomgroup[3]=0
        # Planned optical centers and viewing axes, display-only geometry.
        with viewer_handle.lock():
            scene=viewer_handle.user_scn;scene.ngeom=0
            for item in refs:
                eye=np.array(item['eye_target_world_m']);aim=np.array(item['look_target_world_m'])
                g=scene.geoms[scene.ngeom];scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.007,0.,0.]),eye,np.eye(3).ravel(),np.array([.1,.7,1.,.7]))
                g=scene.geoms[scene.ngeom];scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,.65,1.,.3]))
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.001,eye,aim)
    else:
        renderer=mujoco.Renderer(m,height=512,width=612)
    wall=time.monotonic();last_sync=0.;steps=0;stopped=False;max_pose_error=0.;capture_velocity=0.;recording=True
    acquisition=robot.profile['simulation']['assumed_acquisition_s']
    def advance(q,v,phase,view):
        nonlocal last_sync,steps,stopped
        speed,omega=robot.step(q,v)
        if steps%10==0 and recording:
            quat=np.zeros(4);mujoco.mju_mat2Quat(quat,d.site_xmat[robot.sid])
            logs.append([d.time,view,phase,*q,*v,*d.qpos,*d.qvel,*d.site_xpos[robot.sid],*quat])
        if viewer_handle and d.time-last_sync>=1/60:
            if not viewer_handle.is_running():stopped=True;raise KeyboardInterrupt
            viewer_handle.sync();last_sync=d.time
            target_wall=wall+d.time/rate
            time.sleep(max(0.,target_wall-time.monotonic()))
        steps+=1
        if not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qvel))>20:raise RuntimeError('Unstable state')
        if robot.collision_pairs():raise RuntimeError(f'Contact detected: {robot.collision_pairs()}')
        if np.any(d.qpos<m.jnt_range[:,0]-1e-5) or np.any(d.qpos>m.jnt_range[:,1]+1e-5):raise RuntimeError('Joint limit violation')
        return speed,omega
    try:
        qprev=np.array(refs[0]['q_rad'])
        for view in refs:
            qtarget=np.array(view['q_rad']);duration=view['transition_duration_s']
            if duration>0:
                begin=d.time
                while d.time-begin<duration:
                    u=min((d.time-begin)/duration,1.);s=10*u**3-15*u**4+6*u**5
                    ds=(30*u**2-60*u**3+30*u**4)/duration
                    advance(qprev+s*(qtarget-qprev),ds*(qtarget-qprev),'move',view['id'])
            begin=d.time;stable=0.
            while d.time-begin<5.:
                speed,omega=advance(qtarget,np.zeros(6),'settle',view['id'])
                acceptable=(speed<.0002 and omega<.0005 and np.max(np.abs(d.qvel))<.001 and np.max(np.abs(d.qpos-qtarget))<.001)
                stable=stable+m.opt.timestep if acceptable else 0.
                if d.time-begin>=.4 and stable>=.15:break
            else:raise RuntimeError(f'View {view["id"]} failed to settle')
            settle_s=d.time-begin;trigger_time=d.time;eye=d.site_xpos[robot.sid].copy();R=d.site_xmat[robot.sid].reshape(3,3).copy()
            capture_v=0.;capture_w=0.
            while d.time-trigger_time<acquisition:
                speed,omega=advance(qtarget,np.zeros(6),'capture_hold',view['id'])
                capture_v=max(capture_v,speed);capture_w=max(capture_w,omega)
            drift=np.linalg.norm(d.site_xpos[robot.sid]-eye);angle=np.linalg.norm(angle_error(R,d.site_xmat[robot.sid].reshape(3,3)))
            pixel_bound=robot.profile['simulation']['fx']*(drift/.3+angle)
            if capture_v>.0002 or capture_w>.0005:raise RuntimeError('Camera moved beyond settle criteria during capture hold')
            seen,local=visible(robot,points);observed|=seen
            ranges=local[seen,2]
            T=np.eye(4);T[:3,:3]=d.site_xmat[robot.sid].reshape(3,3);T[:3,3]=d.site_xpos[robot.sid]
            max_pose_error=max(max_pose_error,float(np.linalg.norm(T[:3,3]-view['eye_target_world_m'])))
            per={**view,'trigger_time_s':float(trigger_time),'hold_end_time_s':float(d.time),'settle_s':float(settle_s),
                 'actual_T_world_camera':T.tolist(),'actual_q_rad':d.qpos.tolist(),
                 'max_capture_linear_speed_m_s':capture_v,'max_capture_angular_speed_rad_s':capture_w,
                 'acquisition_drift_m':float(drift),'acquisition_rotation_rad':float(angle),'geometric_pixel_drift_bound':float(pixel_bound),
                 'visible_seam_samples':int(seen.sum()),'total_seam_samples':len(points),
                 'visible_seam_ids':sorted(set(ids[seen].tolist())),
                 'visible_depth_range_m':[float(ranges.min()),float(ranges.max())] if len(ranges) else None}
            perview.append(per);triggers.append([view['id'],trigger_time,d.time,acquisition])
            print(f'{part}: view {view["id"]}, stop {settle_s:.2f}s, hold {acquisition:.2f}s, visible weld samples {seen.sum()}/{len(points)}',flush=True)
            if renderer:
                option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
                renderer.update_scene(d,camera='scan_camera',scene_option=option)
                png(out/f'camera_view_{view["id"]:02d}.png',renderer.render())
            qprev=qtarget
        coverage={seam:float(observed[ids==seam].mean()) for seam in sorted(set(ids))}
        warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number}
        summary={'part':part,'passed':bool(np.all(observed) and robot.contacts==0 and robot.saturated==0 and not warnings and robot.max_v<=.08),
                 'scope':'Camera motion, visibility and capture trigger poses only; no image processing or point-cloud generation',
                 'views':len(refs),'duration_s':float(d.time),'physics_steps':steps,
                 'seam_sample_coverage':float(observed.mean()),'coverage_by_seam_id':coverage,'total_seam_samples':len(points),
                 'visibility_test':'Known seam points every <=2 mm; nominal FOV/range and triangle-mesh ray occlusion within 2 mm. Geometric check only.',
                 'maximum_camera_move_speed_m_s':robot.max_v,'maximum_camera_angular_speed_rad_s':robot.max_omega,
                 'max_camera_pose_error_m':max_pose_error,'robot_contact_steps':robot.contacts,'torque_saturated_steps':robot.saturated,
                 'max_torque_Nm':robot.max_tau.tolist(),'warnings':warnings,
                 'viewpoints':perview,'limitations':'No optical quality prediction. Nominal camera FOV/weight, assumed mount/COM/inertia/controller and acquisition hold; known static fixture; sampled collision validation.'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'capture_waypoints.json').write_text(json.dumps({'camera_profile':robot.profile,'views':perview},indent=2))
        fields=['time_s','view_id','phase']+[f'{name}{j}' for name in ['q_ref_rad_','qdot_ref_rad_s_','q_actual_rad_','qdot_actual_rad_s_'] for j in range(1,7)]+['camera_x_m','camera_y_m','camera_z_m','qw','qx','qy','qz']
        with (out/'joint_trajectory.csv').open('w') as f:w=csv.writer(f);w.writerow(fields);w.writerows(logs)
        with (out/'capture_triggers.csv').open('w') as f:w=csv.writer(f);w.writerow(['view_id','trigger_time_s','hold_end_s','assumed_acquisition_s']);w.writerows(triggers)
        print(json.dumps({k:v for k,v in summary.items() if k!='viewpoints'},indent=2),flush=True)
        if not summary['passed']:raise RuntimeError('Camera trajectory acceptance failed; inspect summary')
        if viewer_handle:
            if loop:
                recording=False
                replay_t=np.array([row[0] for row in logs])
                replay_q=np.array([row[3:9] for row in logs])
                replay_v=np.array([row[9:15] for row in logs])
                print('Scan completed. Loop enabled: return along the checked path, then scan again.',flush=True)
                cycle=1
                while viewer_handle.is_running():
                    for direction,label in [(-1,'return'),(1,'scan')]:
                        # Keep integrating torque-controlled physics. No qpos reset.
                        hold_start=d.time
                        while d.time-hold_start<2.:
                            advance(qprev,np.zeros(6),'end_hold',-1)
                        print(f'Cycle {cycle}: {label}',flush=True)
                        begin=d.time;last_status=-1.
                        while d.time-begin<replay_t[-1]:
                            elapsed=d.time-begin
                            phase_t=replay_t[-1]-elapsed if direction<0 else elapsed
                            qref=np.array([np.interp(phase_t,replay_t,replay_q[:,j]) for j in range(6)])
                            vref=direction*np.array([np.interp(phase_t,replay_t,replay_v[:,j]) for j in range(6)])
                            advance(qref,vref,label,-1)
                            if elapsed-last_status>=.5:
                                (out/'live_status.json').write_text(json.dumps({'cycle':cycle,'phase':label,'phase_time_s':elapsed,
                                    'sim_time_s':float(d.time),'q_actual_rad':d.qpos.tolist(),'joint_speed_rad_s':d.qvel.tolist()},indent=2))
                                last_status=elapsed
                        qprev=replay_q[0 if direction<0 else -1]
                    cycle+=1
            print('Scan completed; holding final pose. Close window to exit.',flush=True)
            while viewer_handle.is_running():
                for _ in range(8):robot.step(qprev,np.zeros(6))
                viewer_handle.sync();time.sleep(.016)
    finally:
        if renderer:renderer.close()
        if viewer_handle:viewer_handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['tee','v_plate','all'],default='tee');p.add_argument('--view',action='store_true');p.add_argument('--rate',type=float,default=1.)
    p.add_argument('--loop',action='store_true',help='Viewer: physically return along the checked path and repeat')
    a=p.parse_args()
    if a.rate<=0:raise SystemExit('rate must be positive')
    from export_handoff import export
    for part in ['tee','v_plate'] if a.part=='all' else [a.part]:
        run(part,a.view,a.rate,a.loop)
        if not a.view:
            export(part)
