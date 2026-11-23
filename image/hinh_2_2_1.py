#!/usr/bin/env python3
"""Hình 2.2.1. Sơ đồ khâu–khớp SR5 với các trục x và z."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trajectory' / 'v_path'))
from analyze import Robot

INK='#203345';BLUE='#176eae';RED='#c44743';GREY='#8295a4'
COLORS=['#70869a','#8d63b8','#277db8','#159587','#dc9540','#7857aa','#42556c']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'svg.fonttype':'none'})


def main():
    robot=Robot();qdeg=np.array([0,30,-60,0,-90,0]);_,_,links=robot.fk_jacobian(np.deg2rad(qdeg))
    centers=np.array([links[f'xMateSR5_link{i}'][:3,3]*1000 for i in range(1,7)])
    axes=[]
    for joint in robot.chain:
        if joint.get('type')=='revolute':
            parent=links[joint.find('parent').get('link')]
            axes.append(parent[:3,:3]@np.fromstring(joint.find('axis').get('xyz'),sep=' '))
    axes=np.array(axes)
    assert np.allclose(np.linalg.norm(axes,axis=1),1)
    assert np.allclose(centers[0],centers[1])
    h=np.array([np.sqrt(3)/2,.5,0]);depth=np.array([-.5*np.cos(.45),np.sqrt(3)/2*np.cos(.45),-np.sin(.45)])
    B=np.vstack([h,np.cross(h,depth),depth])
    def p(v):return np.asarray(v)@B.T
    fig=plt.figure(figsize=(15,12),facecolor='white')
    ax=fig.add_axes([.02,.09,.96,.89]);ax.set_aspect('equal');ax.axis('off')
    ax.set_xlim(-240,850);ax.set_ylim(-75,850)
    def line(a,b,color,lw=5,style='-',z=3):
        pts=p(np.array([a,b]));ax.plot(pts[:,0],pts[:,1],color=color,lw=lw,ls=style,solid_capstyle='round',zorder=z)
    def label(point,text,xy,color=INK):
        ax.annotate(text,p(point)[:2],xytext=xy,ha='left',va='center',fontsize=11,color=color,
            bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',color=color,lw=.85),zorder=15)
    def dimension(a,b,offset,text):
        a=p(a)[:2];b=p(b)[:2];d=b-a;n=np.array([-d[1],d[0]])/np.linalg.norm(d)
        aa=a+n*offset;bb=b+n*offset
        for v,w in [(a,aa),(b,bb)]:ax.plot(*np.array([v,w]).T,color=GREY,lw=.7,ls=(0,(3,3)),zorder=1)
        ax.annotate('',bb,aa,arrowprops=dict(arrowstyle='<->',color=BLUE,lw=1),zorder=2)
        angle=np.degrees(np.arctan2(d[1],d[0]));angle=(angle+90)%180-90
        xy=(aa+bb)/2+n*np.sign(offset)*16
        ax.text(*xy,text,ha='center',va='center',rotation=angle,rotation_mode='anchor',color=BLUE,
            fontsize=12,fontweight='bold',bbox=dict(fc='white',ec='none',pad=1.5),zorder=14)
    def revolute(center,axis,index):
        # Cylinder and arc lie in planes normal to the actual positive joint axis.
        helper=np.array([1.,0,0]) if abs(axis[0])<.8 else np.array([0.,1,0])
        u=np.cross(axis,helper);u/=np.linalg.norm(u);v=np.cross(axis,u)
        radius=20 if index!=0 else 16;half=35 if index!=0 else 28
        t=np.linspace(0,2*np.pi,49)
        rings=[center+s*half*axis+radius*(np.cos(t)[:,None]*u+np.sin(t)[:,None]*v) for s in [-1,1]]
        # A projected convex cylinder silhouette, then the two rim ellipses.
        pts=p(np.vstack(rings))[:,:2]
        from scipy.spatial import ConvexHull
        hull=ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices],fc='#eef3f7',ec=COLORS[index+1],lw=1.4,zorder=6+index*.05))
        for ring in rings:
            xy=p(ring);ax.plot(xy[:,0],xy[:,1],color=COLORS[index+1],lw=.9,zorder=7+index*.05)
    J1,J2,J3,J4,J5,J6=centers
    upper=J2+links['xMateSR5_link2'][:3,:3]@np.array([0,0,400])
    fore=J3+links['xMateSR5_link3'][:3,:3]@np.array([0,0,400])
    base=np.zeros(3)
    line(base,J1,COLORS[0],8)
    line(J2,upper,COLORS[2],8);line(upper,J3,COLORS[2],8)
    line(J3,fore,COLORS[3],8);line(fore,J4,COLORS[3],8)
    line(J4,J5,COLORS[4],8);line(J5,J6,COLORS[5],8)
    # Fixed base symbol at world origin.
    O=p(base)[:2]
    ax.add_patch(Polygon([O+[-48,0],O+[48,0],O+[68,-24],O+[-68,-24]],fc='#e1e8ee',ec=INK,lw=1,zorder=5))
    for xx in np.arange(-58,65,18):ax.plot([xx,xx-14],[-26,-43],color=GREY,lw=.7)

    # Place the J1 symbol near the base, along the actual joint axis.
    symbolic=base+np.array([0,0,32])
    for i,(center,axis) in enumerate(zip(centers,axes)):
        revolute(symbolic if i==0 else center,axis,i)
    dimension(base,J1,140,'328 mm')
    label(symbolic,'J₁',(-108,70),COLORS[1])
    label(J2,'J₂ · O₁ = O₂',(-118,323),INK)
    label(J3,'J₃',(130,610),INK)
    label(J4,'J₄',(470,560),INK)
    label(J5,'J₅',(668,617),INK)
    label(J6,'J₆',(695,435),INK)
    for i in range(7):
        frame=links['xMateSR5_base' if i==0 else f'xMateSR5_link{i}']
        xy=p(frame[:3,3]*1000)[:2]
        ax.scatter(*xy,s=15,color=INK,zorder=17)
        for k,ch,color in [(0,'x',RED),(2,'z',BLUE)]:
            if (i,k)==(2,2):
                # An oblique arrow depicts depth toward the viewer.
                end=xy+np.array([-100,-75])
                ax.annotate('',end,xy,arrowprops=dict(arrowstyle='-|>',lw=2,color=BLUE,mutation_scale=15,shrinkA=0,shrinkB=0),zorder=20)
                ax.text(*(end+[-16,-12]),'z₂',color=BLUE,fontsize=16,weight='bold',ha='center',va='center',zorder=22,path_effects=[pe.withStroke(linewidth=3,foreground='white')])
                continue
            d=p(frame[:3,k])[:2]
            end=xy+d*({4:60,5:52,6:72}.get(i,90))
            prominent=(i,k)==(3,2)
            if prominent:
                color='#103d71'
            arrow=ax.annotate('',end,xy,arrowprops=dict(arrowstyle='-|>',lw=2.8 if prominent else 2,color=color,shrinkA=0,shrinkB=0,mutation_scale=17 if prominent else 15),zorder=16)
            if prominent:
                arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=5.5,foreground='white')])
            ax.text(*(end+d*13+([0,16] if (i,k)==(4,2) else [12,0] if (i,k)==(5,2) else [0,0])),ch+'₀₁₂₃₄₅₆'[i],color=color,fontsize=16,ha='center',va='center',weight='bold',path_effects=[pe.withStroke(linewidth=3,foreground='white')],zorder=18)
    fig.text(.5,.07,'q = (0°, 30°, −60°, 0°, −90°, 0°)',ha='center',fontsize=16)
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    for ext in ('svg','png'):
        fig.savefig(output / f'hinh_2_2_1.{ext}',dpi=200,bbox_inches='tight',pad_inches=.2)


if __name__ == '__main__':
    main()
