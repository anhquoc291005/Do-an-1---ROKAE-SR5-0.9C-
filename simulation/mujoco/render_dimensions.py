#!/usr/bin/env python3
"""Render a dimensioned engineering plate directly from the local SR5 URDF/STLs."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'trajectory/tee'))
from analyze import Robot, origin
from render_scene import stl, transformed

INK = '#192b3b'
BLUE = '#176eae'
RED = '#c3463e'
GREY = '#80919f'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': INK, 'svg.fonttype': 'none'})


def dimv(ax, x, z0, z1, x0, x1, text):
    for z, xx in [(z0, x0), (z1, x1)]:
        ax.plot([xx, x], [z, z], color=GREY, lw=.7, linestyle=(0,(3,3)), zorder=5)
    ax.annotate('', (x,z1), (x,z0), arrowprops=dict(arrowstyle='<->', color=BLUE, lw=1.1), zorder=6)
    ax.text(x-15, (z0+z1)/2, text, rotation=90, va='center', ha='right', color=BLUE,
            fontsize=12, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1.5), zorder=7)


def dimh(ax, z, x0, x1, original_z, text):
    for x in [x0,x1]:
        ax.plot([x,x], [original_z,z], color=GREY, lw=.7, linestyle=(0,(3,3)), zorder=5)
    ax.annotate('', (x1,z), (x0,z), arrowprops=dict(arrowstyle='<->', color=BLUE, lw=1.1), zorder=6)
    ax.text((x0+x1)/2, z-24, text, va='top', ha='center', color=BLUE, fontsize=12,
            fontweight='bold', bbox=dict(fc='white', ec='none', pad=1.5), zorder=7)


def note(ax, point, label, text_xy, color=INK):
    ax.annotate(label, point, xytext=text_xy, fontsize=10, color=color,
                ha='left', va='center', bbox=dict(fc='white', ec='none', pad=1.5),
                arrowprops=dict(arrowstyle='-', color=color, lw=.8), zorder=9)


def main():
    robot=Robot()
    _,_,links=robot.fk_jacobian(np.zeros(6))
    triangles=[]; colors=[]
    light=np.array([-.45,-.65,.8]); light/=np.linalg.norm(light)
    for link in robot.xml.findall('link'):
        name=link.get('name')
        if name=='welding_tcp': continue
        visual=link.find('visual')
        if visual is None: continue
        mesh=visual.find('geometry/mesh')
        if mesh is None: continue
        vertices=stl(mesh.get('filename').removeprefix('file://'))
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        v=transformed(vertices*scale, links[name]@origin(visual.find('origin')))*1000
        normals=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
        shade=.57+.43*np.abs(normals@light)
        base=np.array(to_rgb('#9aaebb' if name=='xMateSR5_base' else '#d5dfe6'))
        triangles.append(v);colors.append(np.clip(shade[:,None]*base,0,1))
    triangles=np.concatenate(triangles);colors=np.concatenate(colors)
    fig=plt.figure(figsize=(18,12),facecolor='white')
    fig.text(.045,.956,'ROKAE xMateSR5', fontsize=26,fontweight='bold')
    fig.text(.045,.923,'KÍCH THƯỚC ĐỘNG HỌC GIỮA CÁC TRỤC KHỚP',fontsize=16,color=BLUE,fontweight='bold')
    fig.text(.955,.953,'Đơn vị: mm\nq₁ = q₂ = q₃ = q₄ = q₅ = q₆ = 0 rad',ha='right',va='top',fontsize=11)
    fig.add_artist(plt.Line2D([.045,.955],[.90,.90],transform=fig.transFigure,color='#d6e1e8',lw=1))
    # Camera bases: horizontal, vertical, and depth (far-to-near sorted below).
    front=np.array([[1,0,0],[0,0,1],[0,1,0]],float)
    side=np.array([[0,1,0],[0,0,1],[-1,0,0]],float)
    h=np.array([1.,-.8,0]);h/=np.linalg.norm(h)
    depth=np.array([.8,1,.24]);depth/=np.linalg.norm(depth)
    up=np.cross(h,depth);up/=np.linalg.norm(up)
    iso=np.vstack([h,up,depth])
    axes=[]
    for i,(basis,title,subtitle) in enumerate([
        (front,'01  HÌNH CHIẾU X–Z','Thấy độ lệch X của hai đoạn 400'),
        (side,'02  HÌNH CHIẾU Y–Z','Thấy độ lệch cổ tay theo Y'),
        (iso,'03  GÓC NHÌN KHÔNG GIAN','Robot trần • không gắn dụng cụ')]):
        ax=fig.add_axes([.04+i*.315,.265,.29,.565])
        projected=triangles@basis.T
        order=np.argsort(-projected[:,:,2].mean(axis=1))
        ax.add_collection(PolyCollection(projected[order,:,:2],facecolors=colors[order],
                          edgecolors='none',rasterized=True,zorder=1))
        ax.set_aspect('equal');ax.set_xlim(-310,310);ax.set_ylim(-80,1340);ax.axis('off')
        ax.text(.5,1.055,title,transform=ax.transAxes,ha='center',fontsize=12,fontweight='bold')
        ax.text(.5,1.02,subtitle,transform=ax.transAxes,ha='center',fontsize=9,color='#5d7384')
        axes.append(ax)
    front_ax,side_ax,iso_ax=axes
    centers=np.array([links[f'xMateSR5_link{i}'][:3,3]*1000 for i in range(1,7)])
    flange=centers[-1]
    assert np.allclose(centers,[[0,0,328],[0,0,328],[50,0,728],[0,0,1128],[0,136,1128],[0,136,1231.5]])
    for ax,basis in zip(axes,[front,side,iso]):
        p=centers@basis.T
        ax.plot(p[:,0],p[:,1],ls=(0,(4,3)),color=BLUE,lw=1,alpha=.75,zorder=3)
        ax.scatter(p[:,0],p[:,1],s=24,facecolors='white',edgecolors=RED,lw=1.2,zorder=8)
        base=np.array([0,0,0])@basis.T
        ax.scatter(base[0],base[1],s=25,color=INK,zorder=8)
    dimv(front_ax,-215,0,328,0,0,'328')
    dimv(front_ax,-215,328,728,0,50,'400')
    dimv(front_ax,-215,728,1128,50,0,'400')
    note(front_ax,(0,328),'J1 / J2',(115,355))
    note(front_ax,(50,728),'J3',(150,700))
    note(front_ax,(0,1128),'J4 / J5',(120,1168))
    note(front_ax,(0,1231.5),'J6 · flange',(95,1285))
    # Small signed offsets are shown with exact extension points and external labels.
    for z,x0,x1,text,label_z in [(728,0,50,'ΔX = +50',835),(1128,50,0,'ΔX = −50',1030)]:
        line_z=z+45 if z==728 else z-45
        for x in [x0,x1]:front_ax.plot([x,x],[z,line_z],color=RED,lw=.8,zorder=5)
        front_ax.annotate('',(x1,line_z),(x0,line_z),arrowprops=dict(arrowstyle='<->',color=RED,lw=1),zorder=6)
        note(front_ax,((x0+x1)/2,line_z),text,(115,label_z),RED)
    side_ax.plot([0,0],[-30,1270],ls=(0,(3,4)),color=BLUE,lw=.8,zorder=2)
    dimh(side_ax,1010,0,136,1128,'136')
    dimv(side_ax,265,1128,1231.5,136,136,'103,5')
    note(side_ax,(0,328),'J1 / J2',(-195,380))
    note(side_ax,(0,728),'J3',(-175,755))
    note(side_ax,(0,1128),'J4',(-185,1180))
    note(side_ax,(136,1128),'J5',(160,1080))
    note(side_ax,(136,1231.5),'J6',(145,1290))
    for ax,hlabel in [(front_ax,'+X'),(side_ax,'+Y')]:
        ax.annotate('',(160,-35),(15,-35),arrowprops=dict(arrowstyle='->',color=INK,lw=1))
        ax.text(170,-35,hlabel,va='center',fontsize=9)
        ax.annotate('',(220,140),(220,20),arrowprops=dict(arrowstyle='->',color=INK,lw=1))
        ax.text(220,155,'+Z',ha='center',fontsize=9)
        ax.text(12,9,'O',fontsize=10,fontweight='bold')
    ip=centers@iso.T
    for j,point in enumerate(ip):
        if j==1:continue
        label='J1 / J2' if j==0 else f'J{j+1}'
        offset=(-150,15) if j in [0,2,3] else (80,-15)
        note(iso_ax,point[:2],label,(point[0]+offset[0],point[1]+offset[1]))
    fig.text(.045,.235,'THÔNG SỐ URDF — tịnh tiến từ khâu cha tới gốc khớp, trong hệ khâu cha',fontsize=11,fontweight='bold')
    table_ax=fig.add_axes([.045,.105,.91,.11]);table_ax.axis('off')
    translations=[];axes_local=[]
    for joint in robot.xml.findall('joint'):
        if joint.get('type')!='revolute':continue
        xyz=np.fromstring(joint.find('origin').get('xyz'),sep=' ')*1000
        translations.append('('+ '; '.join(f'{x:g}'.replace('.',',') for x in xyz)+')')
        axis=joint.find('axis').get('xyz')
        axes_local.append({'0 0 1':'+Z','0 1 0':'+Y','0 -1 0':'−Y'}[axis])
    table=table_ax.table(cellText=[['Δp (X; Y; Z)']+translations,['Trục quay']+axes_local],
        colLabels=['Thông số']+[f'J{i}' for i in range(1,7)],loc='center',cellLoc='center',
        colWidths=[.16]+[.14]*6)
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d6e1e8');cell.set_linewidth(.7)
        cell.set_facecolor('#eaf2f7' if r==0 else '#f8fafc')
        if r==0:cell.set_text_props(weight='bold',color=BLUE)
    fig.text(.045,.067,'Gốc flange tại q = 0: (0; 136; 1.231,5) mm.  •  RPY của 6 joint origin đều bằng 0.',fontsize=10)
    fig.text(.045,.040,'Kích thước theo URDF xMateSR5 của ROKAE đang dùng trong dự án; đây chưa phải bảng DH hoặc kích thước đã đo trên robot thật.',fontsize=9,color='#5d7384')
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        path=out/f'sr5_kinematic_dimensions.{ext}'
        fig.savefig(path,dpi=200,facecolor='white')
        print(path)
    (out/'sr5_kinematic_dimensions.json').write_text(json.dumps({'source_urdf':str(ROOT.parents[1]/'trajectory/tee/generated/sr5_torch.urdf'),
        'q_rad':[0]*6,'joint_origins_world_mm':centers.tolist(),'flange_world_mm':flange.tolist(),
        'note':'URDF geometric parameters; not a DH table or verified hardware metrology.'},indent=2))
    plt.close(fig)


if __name__=='__main__':main()
