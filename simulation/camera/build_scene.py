#!/usr/bin/env python3
"""Industrial camera proxy on the SR5 flange; build independent scan scenes."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as E
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
WIDTH,HEIGHT=612,512
FX,FY=WIDTH*.6/.570,HEIGHT*.6/.460
PROFILE={
    'reference_camera':'Zivid 2+ MR60',
    'source':'https://www.zivid.com/hubfs/User%20guides%20and%20datasheets/Zivid%202%2B%20MR60%20Datasheet.pdf',
    'source_revision':'1.3, July 2025',
    'datasheet':{'technology':'structured light','native_resolution':[2448,2048],
                 'focus_m':.6,'recommended_range_m':[.3,1.1],'optimal_range_m':[.35,.9],
                 'fov_m_at_0_6m':[.570,.460],'housing_size_m':[.169,.124,.056],'mass_kg':1.0,
                 'typical_acquisition_ms':[15,800],'typical_capture_ms':[25,1500]},
    'simulation':{'resolution':[WIDTH,HEIGHT],'fx':FX,'fy':FY,'cx':WIDTH/2,'cy':HEIGHT/2,
                  'model':'ideal symmetric pinhole fitted to nominal FOV at 0.6 m; no distortion',
                  'camera_optical_convention':'x right, y down, z forward',
                  'T_flange_optical':[[1,0,0,0],[0,1,0,0],[0,0,1,.120],[0,0,0,1]],
                  'mount_mass_kg':.15,'nominal_standoff_m':.5,'assumed_acquisition_s':.2,
                  'settle_linear_speed_m_s':.0002,'settle_angular_speed_rad_s':.0005,
                  'settle_joint_speed_rad_s':.001,'settle_continuous_s':.15,
                  'transition_peak_speed_limit_m_s':.08,
                  'limitations':'No structured-light projector visibility, phase decoding, SNR, reflection, motion blur, hardware intrinsic/hand-eye calibration or noise model. Housing/COM/mount are proxies.'}}


def build(part):
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    xml=E.parse(WORKSPACE/'simulation/mujoco/generated/sr5_scene.xml').getroot()
    xml.set('model',f'SR5_flange_camera_scan_{part}')
    world=xml.find('worldbody');asset=xml.find('asset')
    # Separate scan setup: camera replaces the illustrative welding torch.
    flange=xml.find(".//body[@name='xMateSR5_link6']")
    flange.remove(flange.find("body[@name='welding_tcp']"))
    for element in list(world):
        name=element.get('name','')
        if name.startswith(('V_seam_','path_')) or name=='contact_probe':world.remove(element)
    contact=xml.find('contact')
    for element in list(contact):
        if element.get('body2')=='welding_tcp':contact.remove(element)
    cam=E.SubElement(flange,'body',name='industrial_camera',pos='0 0 .080')
    mass=1.;x,y,z=.169,.124,.056
    E.SubElement(cam,'inertial',pos='0 0 0',mass=str(mass),diaginertia=f'{mass*(y*y+z*z)/12} {mass*(x*x+z*z)/12} {mass*(x*x+y*y)/12}')
    E.SubElement(cam,'geom',name='camera_housing',type='box',size='.0845 .062 .028',rgba='.12 .22 .31 1')
    E.SubElement(cam,'geom',name='camera_lens',type='cylinder',pos='0 0 .029',size='.016 .002',contype='0',conaffinity='0',rgba='.15 .7 .9 1')
    mount=E.SubElement(flange,'body',name='camera_mount',pos='0 0 .035')
    E.SubElement(mount,'inertial',pos='0 0 0',mass='.15',diaginertia='.00012 .00012 .00012')
    E.SubElement(mount,'geom',name='camera_mount_geom',type='cylinder',size='.025 .025',rgba='.3 .3 .33 1',contype='0',conaffinity='0')
    E.SubElement(cam,'site',name='camera_optical',pos='0 0 .040',size='.003',rgba='.1 .8 1 1')
    E.SubElement(cam,'camera',name='scan_camera',pos='0 0 .040',quat='0 1 0 0',
                 resolution=f'{WIDTH} {HEIGHT}',sensorsize='.0057 .0046',focal='.006 .006')
    E.SubElement(contact,'exclude',body1='xMateSR5_link6',body2='industrial_camera')
    E.SubElement(contact,'exclude',body1='camera_mount',body2='industrial_camera')
    source=WORKSPACE/('trajectory/tee' if part=='tee' else 'trajectory/v_path')/'generated'
    scene=json.loads((source/'scene.json').read_text())
    asset.find("mesh[@name='workpiece']").set('file',str(source/'workpiece_mm.stl'))
    # Move the fixture 100 mm towards the base so the far-side camera pose is reachable.
    shift=np.array([-.1,0.,0.]);part_origin=np.array(scene['work_origin_m'])+shift
    work=world.find("geom[@name='workpiece_visual']");work.set('pos',' '.join(map(str,part_origin)))
    world.remove(world.find("geom[@name='workpiece_collision']"))
    E.SubElement(world,'geom',name='part_collision',type='mesh',mesh='workpiece',pos=' '.join(map(str,part_origin)),group='3',rgba='.4 .5 .6 0')
    table=world.find("geom[@name='table']")
    table.set('pos',' '.join(map(str,np.array(scene['table']['center'])+shift)))
    table.set('size',' '.join(map(str,np.array(scene['table']['size'])/2)))
    if part=='tee':
        labels=json.loads((source/'p0000_coupon_tee_welds.json').read_text())
        seams=[np.array(s['polyline'])*.001+part_origin for s in labels['candidate_seams'] if s['is_candidate_weld']]
    else:
        pts=np.array(scene['seam_endpoints_world_m'])+shift
        pts[:,2]-=.0025
        seams=[pts[:2],pts[1:]]
    for i,path in enumerate(seams):
        for j,(a,b) in enumerate(zip(path[:-1],path[1:])):
            E.SubElement(world,'site',name=f'weld_{i}_{j}',type='capsule',fromto=' '.join(map(str,np.r_[a,b])),size='.0015',rgba='1 .18 .1 1',group='2')
    E.SubElement(xml.find('visual'),'map',znear='.001',zfar='10')
    E.indent(xml);path=out/f'{part}_scene.xml';E.ElementTree(xml).write(path,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(path))
    if model.nv!=6 or model.nu!=6:raise RuntimeError('Expected six robot DOFs/motors')
    report={'part':part,'source_mesh':str(source/'workpiece_mm.stl'),'source_scene':str(source/'scene.json'),
            'work_origin_m':part_origin.tolist(),'shift_from_welding_demo_m':shift.tolist(),
            'base_model':'simulation/mujoco/generated/sr5_scene.xml','camera_profile':PROFILE,
            'scan_setup':'Camera and generic mount on flange; welding torch removed for scanning',
            'contact_approximation':'One convex hull for each collision mesh, including phôi; visual geometry remains concave',
            'gravity_m_s2':model.opt.gravity.tolist()}
    (out/f'{part}_scene_report.json').write_text(json.dumps(report,indent=2))
    (out/'camera_profile.json').write_text(json.dumps(PROFILE,indent=2))
    print(f'Built {part}: {path}, optical mount +120 mm, camera 1 kg + mount 0.15 kg')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['tee','v_plate','all'],default='all');args=p.parse_args()
    for part in ['tee','v_plate'] if args.part=='all' else [args.part]:build(part)
