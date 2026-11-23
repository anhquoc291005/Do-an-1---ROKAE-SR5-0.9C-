"""Explain joint 2/3 standard-DH zero offsets using exact frame directions.

The left panel places actual DH origins in the base XZ plane. The right
panels translate direction vectors to a shared point for angle comparison;
they do not redefine the physical origins of the DH frames.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from draw_standard_dh import dh_frames, BETA

OUT=Path(__file__).resolve().parent/'results'
INK='#23394a'; RED='#cf4d43'; BLUE='#227db2'; PURPLE='#7c4ab1'
GREEN='#188c65'; GRAY='#7e909c'; PALE='#d6e0e7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'text.color':INK,'svg.fonttype':'none'})


def vec_arrow(ax, origin, vector, color, width=2., label=None, label_shift=(0,0)):
    o=np.asarray(origin);end=o+np.asarray(vector)
    ax.annotate('',end,o,arrowprops=dict(arrowstyle='-|>',color=color,lw=width,mutation_scale=14),zorder=8)
    if label:
        ax.annotate(label,end,xytext=label_shift,textcoords='offset points',fontsize=13,color=color,
                    ha='left',va='center',zorder=20,bbox=dict(fc='white',ec='none',pad=1))


def arc(ax, start, end, radius, color, label=None, label_angle=None):
    angles=np.linspace(start,end,100)
    pts=radius*np.column_stack([np.cos(angles),np.sin(angles)])
    ax.plot(pts[:,0],pts[:,1],color=color,lw=1.8,zorder=7)
    ax.annotate('',pts[-1],pts[-5],arrowprops=dict(arrowstyle='-|>',color=color,lw=1.8,mutation_scale=12),zorder=9)
    if label:
        angle=(start+end)/2 if label_angle is None else label_angle
        point=(radius+.10)*np.array([np.cos(angle),np.sin(angle)])
        ax.text(*point,label,color=color,fontsize=12,ha='center',va='center',
                bbox=dict(fc='white',ec='none',pad=1),zorder=15)


def normal_symbol(ax, center, into, radius=.045):
    center=np.asarray(center)
    ax.add_patch(Circle(center,radius,fc='white',ec=BLUE,lw=1.4,zorder=25))
    if into:
        for sign in [-1,1]:
            v=radius*.55*np.array([1,sign])
            ax.plot([center[0]-v[0],center[0]+v[0]],
                    [center[1]-v[1],center[1]+v[1]],color=BLUE,lw=1.2,zorder=26)
    else:
        ax.add_patch(Circle(center,radius*.25,fc=BLUE,ec='none',zorder=26))


def signed_angle(u,v,z):
    return np.arctan2(z@np.cross(u,v),u@v)


def angle_panel(ax,joint,qdeg):
    q=np.zeros(6);q[joint-1]=np.deg2rad(qdeg)
    frames=dh_frames(q)
    before=frames[joint-1][:3,0]
    after=frames[joint][:3,0]
    axis=frames[joint-1][:3,2]
    theta=signed_angle(before,after,axis)
    assert np.isclose(theta,np.deg2rad(qdeg)-BETA,atol=1e-12)
    # q1=0 makes XZ a valid, undistorted plane for both angle panels.
    p0,p1=before[[0,2]],after[[0,2]]
    assert np.isclose(np.linalg.norm(p0),1.) and np.isclose(np.linalg.norm(p1),1.)
    ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-.38,1.54);ax.set_ylim(-.32,1.45)
    ax.set_title(f'KHỚP {joint} · q{joint} = {qdeg}°',fontsize=13,weight='bold',loc='left',pad=4)
    vec_arrow(ax,[0,0],1.13*p0,GRAY,label=f'$x_{joint-1}$',label_shift=(5,0))
    vec_arrow(ax,[0,0],1.13*p1,RED,width=2.6,label=f'$x_{joint}$',label_shift=(5,0))
    a0,a1=np.arctan2(p0[1],p0[0]),np.arctan2(p1[1],p1[0])
    arc(ax,a0,a1,.45,PURPLE)
    # The numerical angle is signed about the actual joint axis, not the page normal.
    ax.text(1.00,1.27,f'θ{joint} = {np.rad2deg(theta):.3f}°',color=PURPLE,fontsize=13,
            ha='center',bbox=dict(boxstyle='round,pad=.35',fc='#f4eff9',ec='none'))
    if qdeg:
        qzero=np.zeros(6);Fzero=dh_frames(qzero)
        rest=Fzero[joint][:3,0][[0,2]]
        ax.plot([0,1.10*rest[0]],[0,1.10*rest[1]],color=PALE,lw=1.8,ls='--',zorder=1)
        az=np.arctan2(rest[1],rest[0])
        arc(ax,az,a1,.91,GREEN,label=f'$q_{joint}=30°$')
    normal_symbol(ax,[0,0],into=joint==2)
    ax.text(-.03,-.19,('$z_1=+Y$ : vào trang' if joint==2 else '$z_2=-Y$ : ra khỏi trang'),
            color=BLUE,fontsize=10,ha='left')
    return {'joint':joint,'q_degrees':qdeg,'theta_degrees':float(np.rad2deg(theta)),
            'before_x_base':before.tolist(),'after_x_base':after.tolist(),'positive_axis_base':axis.tolist()}


def geometry_panel(ax):
    F=dh_frames(np.zeros(6))
    o1,o2,o3=[F[i][[0,2],3]*1000 for i in [1,2,3]]
    # Rebase the vertical coordinate to O1 for this shoulder/elbow-only sketch.
    o1-=np.array([0,328]);o2-=np.array([0,328]);o3-=np.array([0,328])
    ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-130,245);ax.set_ylim(-70,610)
    ax.set_title('HÌNH HỌC KHI q₂ = q₃ = 0°',fontsize=13,weight='bold',loc='left',pad=16)
    ax.plot([o1[0],o2[0]],[o1[1],o2[1]],color='#758f9e',lw=10,alpha=.55,solid_capstyle='round',zorder=2)
    ax.plot([0,0,50],[0,400,400],color=PALE,lw=1.2,ls='--',zorder=1)
    x2=F[2][[0,2],0]
    ax.plot([o1[0],o2[0]+130*x2[0]],[o1[1],o2[1]+130*x2[1]],color='#97a9b5',lw=1,ls='--')
    vec_arrow(ax,o1,[150,0],GRAY,label='$x_1$',label_shift=(6,0))
    vec_arrow(ax,o2,130*x2,RED,width=2.5,label='$x_2$',label_shift=(5,2))
    vec_arrow(ax,o3,[180,0],RED,label='$x_3$',label_shift=(5,0))
    for point,name,shift in [(o1,'$O_1$ / J2',(-58,-22)),(o2,'$O_2$ / J3',(12,-21)),(o3,'$O_3$',(-38,17))]:
        ax.scatter(*point,s=65,marker='s' if name=='$O_3$' else 'o',fc='white',ec=INK,lw=1.2,zorder=12)
        ax.annotate(name,point,xytext=shift,textcoords='offset points',fontsize=12,
                    bbox=dict(fc='white',ec='none',pad=1.3),zorder=20)
    # Undistorted geometric beta at O1; its sign as a DH angle depends on z1.
    arc(ax,0,BETA,82,PURPLE)
    ax.text(68,62,'β',color=PURPLE,fontsize=17)
    for start,end,text,pos in [([0,463],[50,463],'50 mm',(25,480)),
                               ([-75,0],[-75,400],'400 mm',(-95,200))]:
        ax.annotate('',end,start,arrowprops=dict(arrowstyle='<->',color=GRAY,lw=1))
        ax.text(*pos,text,fontsize=11,color=GRAY,ha='center',va='center',
                rotation=90 if text=='400 mm' else 0,bbox=dict(fc='white',ec='none',pad=1))
    ax.text(-120,575,'Mặt phẳng XZ, không chiếu xiên',color=GRAY,fontsize=10)
    ax.text(-120,-65,'O₃ là gốc DH trên trục J4 kéo dài.',fontsize=10,color=GRAY)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(18,11.5),facecolor='white')
    fig.text(.045,.95,'SR5 · MINH HỌA RIÊNG KHỚP 2 VÀ KHỚP 3',fontsize=24,weight='bold')
    fig.text(.045,.911,'Cùng một chuyển động vật lý, hai mốc đo góc: q theo URDF và θ theo DH.',fontsize=12,color=BLUE)
    fig.add_artist(plt.Line2D([.045,.96],[.885,.885],transform=fig.transFigure,color=PALE,lw=1))
    geometry_panel(fig.add_axes([.045,.24,.30,.61]))
    results=[]
    for joint,q,box in [(2,0,[.40,.57,.24,.27]),(2,30,[.71,.57,.24,.27]),
                         (3,0,[.40,.22,.24,.27]),(3,30,[.71,.22,.24,.27])]:
        results.append(angle_panel(fig.add_axes(box),joint,q))
    fig.text(.40,.865,'So sánh hướng: các vector x được dịch song song về cùng gốc để đo góc.',fontsize=10,color=GRAY)
    fig.text(.40,.525,'Khớp 2: +z₁ hướng vào trang → q₂ dương quay thuận chiều kim đồng hồ.',fontsize=10,color=BLUE)
    fig.text(.40,.175,'Khớp 3: +z₂ hướng ra khỏi trang → q₃ dương quay ngược chiều kim đồng hồ.',fontsize=10,color=BLUE)
    fig.text(.045,.133,r'$\beta=\operatorname{atan2}(400,50)\approx82.875^\circ$',fontsize=15,color=INK)
    fig.text(.41,.133,r'$\theta_2=q_2-\beta$',fontsize=19,color=PURPLE)
    fig.text(.72,.133,r'$\theta_3=q_3-\beta$',fontsize=19,color=PURPLE)
    fig.text(.045,.076,'Đỏ: hướng x đang xét   •   Xám: hướng tham chiếu   •   Tím: góc DH   •   Xanh lá: góc quay q từ mốc URDF',fontsize=11)
    fig.text(.045,.035,'−β là offset của mốc góc; robot không thực hiện thêm một lượt quay. Các hình góc chỉ so sánh hướng, không đổi gốc hệ DH thực.',
             fontsize=10,color=GRAY)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'sr5_joint23_dh_offsets.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    report={'scope':'Direction-only angle panels; other joints zero; geometry panel in base XZ plane',
            'beta_degrees':float(np.rad2deg(BETA)),'cases':results,'passed':True}
    (OUT/'sr5_joint23_dh_offsets.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
