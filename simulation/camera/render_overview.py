#!/usr/bin/env python3
"""Render the camera and planned view axes, without generating point clouds."""
import argparse
import json
import numpy as np
import mujoco
from robot import Robot,ROOT
from scan_path import png


def main(part):
    robot=Robot(part)
    summary=json.loads((ROOT/'results'/part/'summary.json').read_text())
    chosen=summary['viewpoints'][4]
    robot.reset(np.array(chosen['actual_q_rad']))
    renderer=mujoco.Renderer(robot.model,height=800,width=1280)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.30,0,.48]
    camera.distance=1.6;camera.azimuth=135;camera.elevation=-25
    option=mujoco.MjvOption();option.geomgroup[3]=0
    renderer.update_scene(robot.data,camera=camera,scene_option=option)
    scene=renderer.scene
    for view in summary['viewpoints']:
        eye=np.array(view['eye_target_world_m']);aim=np.array(view['look_target_world_m'])
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.007,0.,0.]),eye,np.eye(3).ravel(),np.array([.1,.7,1.,1.]))
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,.6,.9,.35]))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.0007,eye,aim)
    current_eye=robot.data.site_xpos[robot.sid];R=robot.data.site_xmat[robot.sid].reshape(3,3)
    z=.36;half_x=z*.570/.6/2;half_y=z*.460/.6/2
    corners=[current_eye+R@np.array([x,y,z]) for x,y in [(-half_x,-half_y),(half_x,-half_y),(half_x,half_y),(-half_x,half_y)]]
    for a,b in [(current_eye,c) for c in corners]+[(corners[i],corners[(i+1)%4]) for i in range(4)]:
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,.72,.1,.8]))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.001,a,b)
    path=ROOT/'results'/part/'camera_motion_overview.png'
    png(path,renderer.render());renderer.close();print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['tee','v_plate'],default='tee');a=p.parse_args();main(a.part)
