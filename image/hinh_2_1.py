#!/usr/bin/env python3
"""Hình 2.1. Kích thước hình học và góc khớp của robot SR5."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trajectory' / 'v_path'))
from analyze import Robot,origin
from render_scene import stl,transformed

INK='#192b3b';BLUE='#176eae';RED='#c3463e';GREY='#7f94a3'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'svg.fonttype':'none'})


def main():
    qdeg=np.array([0,30,-60,0,-90,0])
    robot=Robot();_,_,links=robot.fk_jacobian(np.deg2rad(qdeg))
    h=np.array([np.sqrt(3)/2,.5,0])
    depth=np.array([-.5*np.cos(.45),np.sqrt(3)/2*np.cos(.45),-np.sin(.45)])
    B=np.vstack([h,np.cross(h,depth),depth])
    def project(p):return np.asarray(p)@B.T
    vv=[];cc=[]
    light=np.array([.4,-.65,.8]);light/=np.linalg.norm(light)
    for link in robot.xml.findall('link'):
        name=link.get('name')
        if name=='welding_tcp':continue
        visual=link.find('visual');mesh=link.find('visual/geometry/mesh')
        if mesh is None:continue
        vertices=stl(mesh.get('filename').removeprefix('file://'))
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        v=transformed(vertices*scale,links[name]@origin(visual.find('origin')))*1000
        normals=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        shade=.52+.48*np.abs(normals@light)
        rgb=np.array(to_rgb('#8ea4b4' if name=='xMateSR5_base' else '#d4e0e8'))
        vv.append(v);cc.append(shade[:,None]*rgb)
    vertices=project(np.concatenate(vv));colors=np.concatenate(cc)
    order=np.argsort(-vertices[:,:,2].mean(axis=1))
    fig=plt.figure(figsize=(15,12),facecolor='white')
    ax=fig.add_axes([.055,.24,.89,.64]);ax.set_aspect('equal');ax.axis('off')
    ax.set_xlim(-235,815);ax.set_ylim(-55,920)
    ax.add_collection(PolyCollection(vertices[order,:,:2],facecolors=colors[order],edgecolors='none',rasterized=True,zorder=1))
    centers=np.array([links[f'xMateSR5_link{i}'][:3,3]*1000 for i in range(1,7)])
    def note(p,text,xy,color=INK):
        ax.annotate(text,project(p)[:2],xytext=xy,ha='left',va='center',color=color,fontsize=11,
            bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',color=color,lw=.85),zorder=9)
    def dimension(a,b,offset,text):
        a=project(a)[:2];b=project(b)[:2];v=b-a
        n=np.array([-v[1],v[0]])/np.linalg.norm(v)
        a2=a+n*offset;b2=b+n*offset
        for p,pp in [(a,a2),(b,b2)]:ax.plot(*np.array([p,pp]).T,color=GREY,lw=.8,ls=(0,(3,3)),zorder=4)
        ax.annotate('',b2,a2,arrowprops=dict(arrowstyle='<->',color=BLUE,lw=1.3),zorder=5)
        angle=np.rad2deg(np.arctan2(v[1],v[0]))
        if angle>90:angle-=180
        if angle< -90:angle+=180
        textp=(a2+b2)/2+n*np.sign(offset)*16
        ax.text(*textp,text,rotation=angle,rotation_mode='anchor',ha='center',va='center',fontsize=14,
            color=BLUE,fontweight='bold',bbox=dict(fc='white',ec='none',pad=2),zorder=8)
    # Each length is measured along its actual parent-frame component.
    J1,J2,J3,J4,J5,J6=centers
    upper_end=J2+links['xMateSR5_link2'][:3,:3]@np.array([0,0,400])
    fore_end=J3+links['xMateSR5_link3'][:3,:3]@np.array([0,0,400])
    assert np.allclose(upper_end+links['xMateSR5_link2'][:3,:3]@np.array([50,0,0]),J3)
    assert np.allclose(fore_end+links['xMateSR5_link3'][:3,:3]@np.array([-50,0,0]),J4)
    dimension(np.zeros(3),J1,125,'328 mm')
    dimension(J2,upper_end,85,'400')
    dimension(J3,fore_end,90,'400')
    dimension(J4,J5,90,'136')
    dimension(J5,J6,85,'103,5')
    for a,b in [(upper_end,J3),(fore_end,J4)]:
        pair=project(np.array([a,b]));ax.plot(pair[:,0],pair[:,1],color=RED,lw=1.5,zorder=5)
    note((upper_end+J3)/2,'Δx₂ = +50',(330,335),RED)
    note((fore_end+J4)/2,'Δx₃ = −50',(715,675),RED)

    table_ax=fig.add_axes([.055,.125,.89,.075]);table_ax.axis('off')
    table=table_ax.table(cellText=[[f'{v}°' for v in qdeg]],colLabels=[f'q{i}' for i in range(1,7)],
        loc='center',cellLoc='center',colWidths=[1/6]*6)
    table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,1.8)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d6e1e8');cell.set_linewidth(.7);cell.set_facecolor('#eaf2f7' if r==0 else '#f8fafc')
        if r==0:cell.set_text_props(color=BLUE,weight='bold')
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        path = out / f'hinh_2_1.{ext}'
        fig.savefig(path, dpi=200, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
