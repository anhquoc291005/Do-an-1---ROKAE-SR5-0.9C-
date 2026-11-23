#!/usr/bin/python3
"""Publish SR5 joint states and the exact dataset mesh to RViz; visual playback."""
from pathlib import Path
import json
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from analyze import Robot

ROOT = Path(__file__).resolve().parent


class Replay(Node):
    def __init__(self):
        super().__init__('sr5_v_weld_replay')
        self.joints = self.create_publisher(JointState, 'joint_states', 10)
        self.markers = self.create_publisher(MarkerArray, 'welding_scene', 10)
        self.data = np.loadtxt(ROOT/'results/trajectory.csv',delimiter=',',skiprows=1)
        self.scene = json.loads((ROOT/'generated/scene.json').read_text())
        self.robot = Robot()
        self.started = time.monotonic()
        self.rate = float(os.environ.get('WELD_PLAYBACK_RATE','1'))
        if not np.isfinite(self.rate) or self.rate<=0:raise ValueError('Invalid WELD_PLAYBACK_RATE')
        self.create_timer(.02,self.tick)
        self.get_logger().info(f'SR5 V path A-B-C, visual replay x{self.rate:g}; {self.data[-1,0]:.3f} s per pass')

    def tick(self):
        t = min((time.monotonic()-self.started)*self.rate % (self.data[-1,0]+3),self.data[-1,0])
        q = np.array([np.interp(t,self.data[:,0],self.data[:,j]) for j in range(1,7)])
        stamp = self.get_clock().now().to_msg()
        joint = JointState();joint.header.stamp=stamp
        joint.name=[f'joint{i}' for i in range(1,7)];joint.position=q.tolist()
        self.joints.publish(joint)
        array = MarkerArray()
        def marker(idx,kind,color):
            m=Marker();m.header.frame_id='world';m.header.stamp=stamp;m.ns='sr5_weld';m.id=idx
            m.type=kind;m.action=Marker.ADD;m.pose.orientation.w=1.
            m.color.r,m.color.g,m.color.b,m.color.a=[float(c) for c in color]
            array.markers.append(m)
            return m
        mesh=marker(0,Marker.MESH_RESOURCE,(.54,.62,.7,1))
        mesh.mesh_resource=(ROOT/'generated/workpiece_mm.stl').as_uri()
        mesh.scale.x=mesh.scale.y=mesh.scale.z=.001
        mesh.pose.position.x,mesh.pose.position.y,mesh.pose.position.z=[float(x) for x in self.scene['work_origin_m']]
        table=marker(1,Marker.CUBE,(.24,.28,.32,1))
        table.pose.position.x,table.pose.position.y,table.pose.position.z=[float(x) for x in self.scene['table']['center']]
        table.scale.x,table.scale.y,table.scale.z=self.scene['table']['size']
        for idx,key,color in [(2,'seam_endpoints_world_m',(1,.15,.12,1)),(3,'tcp_endpoints_world_m',(1,.75,.05,1))]:
            line=marker(idx,Marker.LINE_STRIP,color);line.scale.x=.002
            line.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in self.scene[key]]
        # Trace completed targets in green so progress is visible on both V legs.
        progress=marker(7,Marker.LINE_STRIP,(.1,1.,.35,1.));progress.scale.x=.003
        targets=np.array(self.scene['tcp_endpoints_world_m'])
        corner=self.scene['corner_time_s'];pause=self.scene['corner_pause_s']
        if t<=corner:
            u=t/corner;s=10*u**3-15*u**4+6*u**5
            done=[targets[0],targets[0]+s*(targets[1]-targets[0])]
        elif t<=corner+pause:done=[targets[0],targets[1]]
        else:
            u=(t-corner-pause)/(self.data[-1,0]-corner-pause);s=10*u**3-15*u**4+6*u**5
            done=[targets[0],targets[1],targets[1]+s*(targets[2]-targets[1])]
        progress.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in done]
        for idx,(label,p) in enumerate(zip('ABC',self.scene['seam_endpoints_world_m']),10):
            mark=marker(idx,Marker.TEXT_VIEW_FACING,(1.,1.,1.,1.));mark.scale.z=.018
            mark.pose.position.x=float(p[0]);mark.pose.position.y=float(p[1]);mark.pose.position.z=float(p[2]+.02)
            mark.text=label
        T,_,_=self.robot.fk_jacobian(q)
        tip=marker(4,Marker.SPHERE,(1,.85,.1,1));tip.scale.x=tip.scale.y=tip.scale.z=.008
        tip.pose.position.x,tip.pose.position.y,tip.pose.position.z=T[:3,3].tolist()
        arrow=marker(5,Marker.ARROW,(.1,.9,.5,1));arrow.scale.x=.003;arrow.scale.y=.008;arrow.scale.z=.012
        arrow.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in (T[:3,3]-.06*T[:3,2],T[:3,3])]
        text=marker(6,Marker.TEXT_VIEW_FACING,(.9,.95,1,1));text.scale.z=.025
        text.pose.position.x=.5;text.pose.position.z=.62
        phase='A to B' if t<self.scene['corner_time_s'] else ('Corner stop' if t<self.scene['corner_time_s']+self.scene['corner_pause_s'] else 'B to C')
        text.text=f'SR5 V path | {phase} | {t:.1f}/{self.data[-1,0]:.1f} s\n{self.scene["seam_length_m"]*1000:.1f} mm | visual replay x{self.rate:g}'
        self.markers.publish(array)


if __name__=='__main__':
    rclpy.init();node=Replay()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
