#!/usr/bin/python3
"""Use the vendor geometry, add an illustrative 180 mm straight torch."""
from pathlib import Path
import xml.etree.ElementTree as E
import yaml
import xacro

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT.parents[1] / 'external/rokae_ros2'
OUT = ROOT / 'generated'
OUT.mkdir(exist_ok=True)
robot = E.fromstring(xacro.process_file(str(VENDOR / 'rokae_description/urdf/xMateSR5.urdf.xacro')).toxml())
# Legacy ROS 1 transmissions are irrelevant to this kinematic experiment;
# transmission 1 even references xMateSR5_joint_7 rather than joint1 upstream.
for element in list(robot.findall('transmission')):
    robot.remove(element)
for mesh in robot.findall('.//mesh'):
    mesh.set('filename', mesh.get('filename').replace('package://rokae_description/', (VENDOR / 'rokae_description').as_uri() + '/'))
tcp = E.SubElement(robot, 'link', name='welding_tcp')
for kind in ('visual', 'collision'):
    item = E.SubElement(tcp, kind)
    E.SubElement(item, 'origin', xyz='0 0 -0.095', rpy='0 0 0')
    E.SubElement(E.SubElement(item, 'geometry'), 'cylinder', radius='0.006', length='0.15')
    if kind == 'visual':
        E.SubElement(E.SubElement(item, 'material', name='torch_copper'), 'color', rgba='0.95 0.42 0.10 1')
joint = E.SubElement(robot, 'joint', name='flange_to_tcp', type='fixed')
E.SubElement(joint, 'parent', link='xMateSR5_link6')
E.SubElement(joint, 'child', link='welding_tcp')
E.SubElement(joint, 'origin', xyz='0 0 0.18', rpy='0 0 0')
semantic = E.fromstring((VENDOR / 'rokae_xMateSR5_moveit_config/config/xMateSR5.srdf').read_text())
semantic.set('name', robot.get('name'))
group = semantic.find("group[@name='rokae_arm']")
for j in list(group.findall('joint')):
    group.remove(j)
group.find('chain').set('tip_link', 'welding_tcp')
# Keep only physical adjacent exclusions; check the vendor's 'Never' pairs too.
for pair in list(semantic.findall('disable_collisions')):
    if pair.get('reason') != 'Adjacent':
        semantic.remove(pair)
E.SubElement(semantic, 'disable_collisions', link1='xMateSR5_link6', link2='welding_tcp', reason='Adjacent')
urdf = E.tostring(robot, encoding='unicode')
srdf = E.tostring(semantic, encoding='unicode')
params = {'robot_description': urdf, 'robot_description_semantic': srdf,
          'robot_description_kinematics': {'rokae_arm': {
              'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
              'kinematics_solver_timeout': 0.05,
              'kinematics_solver_search_resolution': 0.005}}}
(OUT / 'sr5_torch.urdf').write_text(urdf)
(OUT / 'sr5_torch.srdf').write_text(srdf)
(OUT / 'params.yaml').write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}))
print('Generated SR5 + TCP; preserved vendor joint transforms and limits.')
