#!/usr/bin/env python3
"""SR5 joint/link diagram derived from the local URDF, with exact joint axes."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'trajectory/tee'))
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
    fig=plt.figure(figsize=(17,12),facecolor='white')
    fig.text(.045,.953,'SR5 · MÔ HÌNH KHÂU–KHỚP',fontsize=25,fontweight='bold')
    fig.text(.045,.918,'Chuỗi nối tiếp 6 khớp quay (6R) • Tư thế gập • Theo URDF xMateSR5',fontsize=12,color=BLUE)
    fig.text(.96,.951,'Kích thước: mm\nTrục quay theo quy tắc bàn tay phải',ha='right',va='top',fontsize=10)
    fig.add_artist(plt.Line2D([.045,.96],[.893,.893],transform=fig.transFigure,color='#d6e1e8',lw=1))
    ax=fig.add_axes([.035,.23,.675,.645]);ax.set_aspect('equal');ax.axis('off')
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
        radius=20 if index!=0 else 16;half=35 if index!=0 else 58
        t=np.linspace(0,2*np.pi,49)
        rings=[center+s*half*axis+radius*(np.cos(t)[:,None]*u+np.sin(t)[:,None]*v) for s in [-1,1]]
        # A projected convex cylinder silhouette, then the two rim ellipses.
        pts=p(np.vstack(rings))[:,:2]
        from scipy.spatial import ConvexHull
        hull=ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices],fc='#eef3f7',ec=COLORS[index+1],lw=1.4,zorder=6+index*.05))
        for ring in rings:
            xy=p(ring);ax.plot(xy[:,0],xy[:,1],color=COLORS[index+1],lw=.9,zorder=7+index*.05)
        start=center-80*axis;end=center+100*axis
        xy=p(np.array([start,end]))
        ax.plot(xy[:,0],xy[:,1],color=RED,lw=1.1,ls=(0,(4,3)),zorder=9)
        aa=p(center+65*axis)[:2];bb=p(end)[:2]
        ax.annotate('',bb,aa,arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.25),zorder=10)
        t=np.linspace(-.5,3.7,45)
        ring=center+42*(np.cos(t)[:,None]*u+np.sin(t)[:,None]*v)
        xy=p(ring);ax.plot(xy[:,0],xy[:,1],color=RED,lw=1.1,zorder=9)
        ax.annotate('',xy[-1,:2],xy[-4,:2],arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.1),zorder=10)
        cp=p(center);ax.scatter(*cp[:2],s=21,fc='white',ec=INK,lw=1,zorder=12)
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
    for i,(center,axis) in enumerate(zip(centers,axes)):revolute(center,axis,i)
    dimension(base,J1,150,'328')
    dimension(J2,upper,105,'400')
    dimension(J3,fore,110,'400')
    dimension(J4,J5,105,'136')
    dimension(J5,J6,120,'103,5')
    label((upper+J3)/2,'Δx₂ = +50',(300,345),RED)
    label((fore+J4)/2,'Δx₃ = −50',(720,650),RED)
    label(J1+axes[0]*60,'J1 · q₁',(-205,370),COLORS[1])
    label(J2+axes[1]*48,'J2 · q₂',(-205,260),COLORS[2])
    label(J3,'J3 · q₃',(175,660),COLORS[3])
    label(J4,'J4 · q₄',(390,760),COLORS[4])
    label(J5,'J5 · q₅',(735,770),COLORS[5])
    label(J6,'J6 · q₆',(735,400),COLORS[6])
    for a,b,text,color,xy in [(J2,upper,'L2',COLORS[2],(80,460)),(J3,fore,'L3',COLORS[3],(380,430)),
                            (J4,J5,'L4',COLORS[4],(450,345)),(J5,J6,'L5',COLORS[5],(715,510))]:
        label((a+b)/2,text,xy,color)
    label(base,'L0 · đế cố định',(-205,65),COLORS[0])
    label(J6,'L6 · mặt bích',(585,300),COLORS[6])
    # A separate world triad keeps the crossed J1/J2 symbols clear.
    origin2d=O.copy()
    for vec,name,color in [(np.array([100,0,0]),'X₀','#be4949'),(np.array([0,100,0]),'Y₀','#32845d'),(np.array([0,0,100]),'Z₀',BLUE)]:
        endpoint=origin2d+p(vec)[:2]
        ax.annotate('',endpoint,origin2d,arrowprops=dict(arrowstyle='->',color=color,lw=1.4),zorder=13)
        ax.text(*(endpoint+[7,7]),name,color=color,fontsize=10,fontweight='bold')
    ax.text(*(origin2d+[73,-31]),'O₀',fontsize=10)
    # Topology panel: deliberately separates link identity from geometric length.
    side=fig.add_axes([.745,.285,.215,.55]);side.axis('off')
    side.text(0,1,'CẤU TRÚC CHUỖI',fontsize=12,fontweight='bold',va='top')
    side.text(0,.93,'L0 — J1 — L1 — J2 — L2\n     — J3 — L3 — J4 — L4\n     — J5 — L5 — J6 — L6',fontsize=11,linespacing=1.9,va='top')
    side.text(0,.69,'J1…J6: khớp quay\nL0…L6: các khâu cứng',fontsize=11,linespacing=1.8,va='top')
    side.text(0,.55,'QUY ƯỚC',fontsize=12,fontweight='bold',va='top')
    side.text(0,.49,'Đường màu: khâu cứng\nHình trụ: khớp quay\nMũi tên đỏ: trục / chiều quay',fontsize=10,linespacing=1.9,va='top')
    side.text(0,.29,'J1 và J2 có cùng gốc URDF,\nnhưng hai trục vuông góc.\nL1 vẫn là một khâu riêng.',fontsize=10,linespacing=1.7,va='top',
              bbox=dict(boxstyle='round,pad=.65',fc='#eef4f8',ec='none'))
    side.text(0,.08,'Hai đoạn 400 có độ lệch 50:\n|J2J3| = |J3J4| ≈ 403,11 mm.',fontsize=10,linespacing=1.7,va='top')
    fig.text(.045,.22,'GÓC KHỚP VÀ TRỤC DƯƠNG CỤC BỘ THEO URDF',fontsize=11,fontweight='bold')
    ta=fig.add_axes([.045,.104,.915,.105]);ta.axis('off')
    table=ta.table(cellText=[['q (độ)']+[str(x)+'°' for x in qdeg],['Trục cục bộ']+['+Z','+Y','−Y','+Z','−Y','+Z']],
        colLabels=['Thông số']+[f'J{i}' for i in range(1,7)],loc='center',cellLoc='center',colWidths=[.16]+[.14]*6)
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.65)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d6e1e8');cell.set_linewidth(.7);cell.set_facecolor('#eaf2f7' if r==0 else '#f8fafc')
        if r==0:cell.set_text_props(color=BLUE,weight='bold')
    fig.text(.045,.065,'Các đoạn 400 và độ lệch ±50 được vẽ theo phương cục bộ. Hình trụ khớp chỉ là ký hiệu, không phải kích thước cơ khí.',fontsize=10)
    fig.text(.045,.037,'Sơ đồ hình học từ URDF; chưa gán hệ trục DH. Mũi tên trục trên hình đã được quay sang world theo tư thế đang minh họa.',fontsize=10,color='#5d7384')
    out=ROOT/'results'
    out.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        path=out/f'sr5_link_joint_model.{ext}';fig.savefig(path,dpi=200,facecolor='white');print(path)
    (out/'sr5_link_joint_model.json').write_text(json.dumps({'q_degrees':qdeg.tolist(),
        'joint_centers_world_mm':centers.tolist(),'positive_joint_axes_world':axes.tolist(),
        'convention':'URDF joint axes; frame rotations are not a DH assignment'},indent=2))
    plt.close(fig)


if __name__=='__main__':main()
