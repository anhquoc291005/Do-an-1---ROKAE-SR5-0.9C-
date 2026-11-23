#!/usr/bin/env python3
"""Convert the vendor kinematic tree and inertias into an explicit MJCF scene."""
from pathlib import Path
import json
import xml.etree.ElementTree as E
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'trajectory/v_path'
OUT = ROOT / 'generated'
OUT.mkdir(exist_ok=True)


def values(s):
    return np.fromstring(s, sep=' ')


def fmt(v):
    return ' '.join(f'{float(x):.15g}' for x in v)


def main():
    urdf = E.parse(SOURCE / 'generated/sr5_torch.urdf').getroot()
    fixture = json.loads((SOURCE / 'generated/scene.json').read_text())
    xml = E.Element('mujoco', model='ROKAE_SR5_V_path_physics')
    E.SubElement(xml, 'compiler', angle='radian', inertiafromgeom='false', autolimits='true')
    E.SubElement(xml, 'option', timestep='0.002', gravity='0 0 -9.81', integrator='implicitfast', solver='Newton')
    default = E.SubElement(xml, 'default')
    E.SubElement(default, 'joint', damping='0.2', armature='0.01')
    E.SubElement(default, 'geom', friction='0.6 0.005 0.0001', solref='0.01 1', solimp='0.9 0.95 0.001')
    visual = E.SubElement(xml, 'visual')
    E.SubElement(visual, 'global', offwidth='1280', offheight='800')
    E.SubElement(visual, 'headlight', ambient='0.35 0.35 0.35', diffuse='0.7 0.7 0.7')
    E.SubElement(xml, 'statistic', center='0.3 0 0.4', extent='1.3')
    asset = E.SubElement(xml, 'asset')
    E.SubElement(asset,'texture',name='checker',type='2d',builtin='checker',rgb1='.16 .19 .23',rgb2='.23 .27 .31',width='256',height='256')
    E.SubElement(asset,'material',name='floor_mat',texture='checker',texrepeat='8 8',reflectance='.1')
    world = E.SubElement(xml, 'worldbody')
    E.SubElement(world, 'light', pos='0 -1 2', dir='0 0 -1', diffuse='.8 .8 .8')
    E.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .1', material='floor_mat')
    E.SubElement(world,'geom',name='table',type='box',pos=fmt(fixture['table']['center']),size=fmt(np.array(fixture['table']['size'])/2),rgba='.24 .28 .33 1')
    E.SubElement(asset,'mesh',name='workpiece',file=str(SOURCE/'generated/workpiece_mm.stl'),scale='.001 .001 .001')
    E.SubElement(world,'geom',name='workpiece_visual',type='mesh',mesh='workpiece',pos=fmt(fixture['work_origin_m']),contype='0',conaffinity='0',group='1',rgba='.38 .52 .63 1')
    # A solid plate approximation avoids treating a concave mesh as a convex hull.
    E.SubElement(world,'geom',name='workpiece_collision',type='box',pos='.5 0 .296',size='.13 .11 .004',group='3',rgba='.4 .5 .6 0')
    seam=np.array(fixture['seam_endpoints_world_m'])
    for i,(a,b) in enumerate(zip(seam[:-1],seam[1:])):
        a=a+[0,0,.0007];b=b+[0,0,.0007]
        E.SubElement(world,'geom',name=f'V_seam_{i}',type='capsule',fromto=fmt(np.r_[a,b]),size='.0012',contype='0',conaffinity='0',rgba='1 .2 .1 1')
    for name,p in zip('ABC',seam):
        E.SubElement(world,'site',name=f'path_{name}',pos=fmt(p+[0,0,.003]),size='.003',rgba='1 .7 .1 1')
    links={l.get('name'):l for l in urdf.findall('link')}
    children={}
    for j in urdf.findall('joint'):
        children.setdefault(j.find('parent').get('link'),[]).append(j)
    joints=[]
    report={'source':'SR5 URDF inertias, unverified against physical SR5 hardware',
            'mujoco_version':mujoco.__version__,'assumptions':{
                'torch_mass_kg':.35,'torch_radius_m':.006,'torch_length_m':.15,
                'joint_damping_Nms_per_rad':.2,'armature_kg_m2':.01,
                'robot_collision':'one convex hull per vendor collision mesh; adjacent bodies excluded',
                'workpiece_collision':'solid box; the shallow visual V groove is not represented in contact geometry',
                'workpiece_and_table':'fixed to world; no deformation or weld process'},'link_inertias':[]}
    def add_link(parent,name,joint=None):
        link=links[name]
        attrs={'name':name}
        if joint is not None:
            o=joint.find('origin')
            if o is not None:
                attrs.update(pos=o.get('xyz','0 0 0'),euler=o.get('rpy','0 0 0'))
        body=E.SubElement(parent,'body',**attrs)
        if joint is not None and joint.get('type')=='revolute':
            limit=joint.find('limit')
            E.SubElement(body,'joint',name=joint.get('name'),type='hinge',axis=joint.find('axis').get('xyz'),range=f'{limit.get("lower")} {limit.get("upper")}')
            joints.append((joint.get('name'),float(limit.get('effort'))))
        inertial=link.find('inertial')
        if inertial is not None:
            i={k:float(v) for k,v in inertial.find('inertia').attrib.items()}
            I=np.array([[i['ixx'],i['ixy'],i['ixz']],[i['ixy'],i['iyy'],i['iyz']],[i['ixz'],i['iyz'],i['izz']]])
            if np.any(values(inertial.find('origin').get('rpy','0 0 0'))):
                raise ValueError('Nonzero inertial rpy requires explicit tensor rotation')
            eig=np.linalg.eigvalsh(I)
            if eig[0]<=0 or eig[2]>eig[0]+eig[1]:raise ValueError(f'Invalid inertia: {name}')
            E.SubElement(body,'inertial',pos=inertial.find('origin').get('xyz'),mass=inertial.find('mass').get('value'),fullinertia=fmt([i[k] for k in ['ixx','iyy','izz','ixy','ixz','iyz']]))
            report['link_inertias'].append({'link':name,'mass_kg':float(inertial.find('mass').get('value')),'eigenvalues_kg_m2':eig.tolist()})
        elif name=='welding_tcp':
            mass=.35;radius=.006;length=.15
            E.SubElement(body,'inertial',pos='0 0 -.095',mass=str(mass),diaginertia=fmt([mass*(3*radius**2+length**2)/12]*2+[mass*radius**2/2]))
        for kind in ['visual','collision']:
            for k,item in enumerate(link.findall(kind)):
                geometry=item.find('geometry');mesh=geometry.find('mesh');cylinder=geometry.find('cylinder')
                attrs={'name':f'{name}_{kind}_{k}','group':'1' if kind=='visual' else '3',
                       'contype':'0' if kind=='visual' else '1','conaffinity':'0' if kind=='visual' else '1',
                       'rgba':'.8 .84 .88 1' if kind=='visual' else '.7 .2 .2 0'}
                o=item.find('origin')
                if o is not None:attrs.update(pos=o.get('xyz','0 0 0'),euler=o.get('rpy','0 0 0'))
                if mesh is not None:
                    meshname=f'{name}_{kind}_mesh_{k}'
                    E.SubElement(asset,'mesh',name=meshname,file=mesh.get('filename').removeprefix('file://'),scale=mesh.get('scale','1 1 1'))
                    attrs.update(type='mesh',mesh=meshname)
                elif cylinder is not None:
                    attrs.update(type='cylinder',size=f'{cylinder.get("radius")} {float(cylinder.get("length"))/2}')
                    if kind=='visual':attrs['rgba']='.95 .42 .1 1'
                else:raise ValueError(f'Unsupported geom for {name}')
                E.SubElement(body,'geom',**attrs)
        if name=='welding_tcp':E.SubElement(body,'site',name='tcp',size='.004',rgba='.1 1 .3 1')
        for child in children.get(name,[]):add_link(body,child.find('child').get('link'),child)
    for j in children.get('world',[]):add_link(world,j.find('child').get('link'),j)
    contact=E.SubElement(xml,'contact')
    for j in urdf.findall('joint'):
        if j.find('parent').get('link')!='world':
            E.SubElement(contact,'exclude',body1=j.find('parent').get('link'),body2=j.find('child').get('link'))
    actuator=E.SubElement(xml,'actuator')
    for name,effort in joints:
        E.SubElement(actuator,'motor',name=f'{name}_torque',joint=name,gear='1',ctrllimited='true',ctrlrange=fmt([-effort,effort]))
    # Independent contact/gravity probe: a 100 g cube falling onto the plate.
    probe=E.SubElement(world,'body',name='contact_probe',pos='.59 .08 .40')
    E.SubElement(probe,'freejoint',name='probe_free')
    E.SubElement(probe,'inertial',pos='0 0 0',mass='.1',diaginertia='.000006666666667 .000006666666667 .000006666666667')
    E.SubElement(probe,'geom',name='probe_cube',type='box',size='.01 .01 .01',rgba='.95 .65 .1 1')
    E.indent(xml)
    path=OUT/'sr5_scene.xml'
    E.ElementTree(xml).write(path,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(path))
    report.update(nq=model.nq,nv=model.nv,actuators=model.nu,joint_effort_limits=[e for _,e in joints])
    (OUT/'model_report.json').write_text(json.dumps(report,indent=2))
    print(f'MJCF compiled: nq={model.nq}, nv={model.nv}, motors={model.nu}; {path}')


if __name__=='__main__':main()
