"""A verified classic-DH assignment for the local SR5 URDF, with diagrams.

Convention: A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i).
z_(i-1) is the positive physical axis of joint i. Frame 0 equals base;
frame 6 equals the URDF flange, so no fixed endpoint adapters are needed.
"""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parent
from fk import Robot

OUT = ROOT / 'results'
BETA = np.arctan2(.4, .05)
A = np.array([0, np.hypot(.05, .4), -.05, 0, 0, 0])
D = np.array([.328, 0, 0, .4, -.136, .1035])
ALPHA = np.deg2rad([-90, 180, -90, 90, -90, 0])
OFFSET = np.array([0, -BETA, -BETA, 0, 0, 0])
XCOLOR, ZCOLOR, INK = '#ca493e', '#1575b6', '#22374a'
LINKCOLORS = ['#8b9ba6', '#9165b8', '#2e86bd', '#19968b', '#d8983e', '#7664ad']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': INK, 'svg.fonttype': 'none'})
H = np.array([np.sqrt(3)/2, .5, 0.])
DEPTH = np.array([-.5*np.cos(.45), np.sqrt(3)/2*np.cos(.45), -np.sin(.45)])
PROJECTION = np.vstack([H, np.cross(H, DEPTH)])


def dh_frames(q):
    T = np.eye(4)
    frames = [T.copy()]
    for a, d, alpha, theta in zip(A, D, ALPHA, np.asarray(q)+OFFSET):
        c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        T = T @ np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                          [0, sa, ca, d], [0, 0, 0, 1.]])
        frames.append(T.copy())
    return frames


def validate(robot):
    root = robot.xml
    bounds = np.array([[float(root.find(f"joint[@name='joint{i}']/limit").get(k))
                        for k in ['lower', 'upper']] for i in range(1, 7)])
    rng = np.random.default_rng(20261117)
    samples = np.vstack([np.zeros(6), np.deg2rad([0,30,-60,0,-90,0]),
                         rng.uniform(bounds[:, 0], bounds[:, 1], (200, 6))])
    max_p = max_r = max_axis = max_normal = max_line = 0.
    for q in samples:
        _, _, links = robot.fk_jacobian(q)
        frames = dh_frames(q)
        expected = links['xMateSR5_link6']
        max_p = max(max_p, np.linalg.norm(frames[-1][:3, 3]-expected[:3, 3]))
        max_r = max(max_r, np.max(np.abs(frames[-1][:3, :3]-expected[:3, :3])))
        for i in range(1, 7):
            joint = root.find(f"joint[@name='joint{i}']")
            parent = links[joint.find('parent').get('link')]
            axis = parent[:3, :3] @ np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            point = parent[:3, 3] + parent[:3, :3] @ np.fromstring(joint.find('origin').get('xyz'), sep=' ')
            Fp, Fi = frames[i-1], frames[i]
            max_axis = max(max_axis, np.linalg.norm(Fp[:3, 2]-axis))
            max_line = max(max_line, np.linalg.norm(np.cross(Fp[:3, 3]-point, axis)))
            max_normal = max(max_normal, abs(Fi[:3, 0] @ Fp[:3, 2]),
                             np.linalg.norm(Fi[:3, 3]-Fp[:3, 3]-D[i-1]*Fp[:3, 2]-A[i-1]*Fi[:3, 0]))
    report = {'convention': 'Classic/standard DH: Rz(theta) Tz(d) Tx(a) Rx(alpha)',
              'endpoint_frames': 'DH frame 0 = URDF base; DH frame 6 = URDF flange',
              'q_definition': 'URDF joint angles, theta_i=q_i+offset_i; no joint sign flips',
              'configurations_checked': len(samples),
              'max_position_difference_m': float(max_p),
              'max_rotation_matrix_difference': float(max_r),
              'max_positive_joint_axis_difference': float(max_axis),
              'max_joint_axis_line_distance_m': float(max_line),
              'max_common_normal_constraint_residual': float(max_normal)}
    assert max(max_p, max_r, max_axis, max_normal, max_line) < 1e-10, report
    report['passed'] = True
    return report


def project(points):
    return np.asarray(points) @ PROJECTION.T


def segment(ax, p, q, **kwargs):
    xy = project([p, q])
    return ax.plot(xy[:, 0], xy[:, 1], **kwargs)


def arrow(ax, start, end, color, width=1.8):
    ax.annotate('', project(end), project(start),
                arrowprops=dict(arrowstyle='-|>', mutation_scale=12, color=color, lw=width), zorder=12)


def axes_at(ax, F, index, length, labels=True, origin=True, label_shifts=None):
    o, R = F[:3, 3]*1000, F[:3, :3]
    for axis, name, color in [(0, 'x', XCOLOR), (2, 'z', ZCOLOR)]:
        tip = o+length*R[:, axis]
        arrow(ax, o, tip, color)
        if labels:
            xy, direction = project(tip), project(R[:, axis])
            shift = (6 if direction[0] >= 0 else -6, 2)
            if label_shifts:
                shift = label_shifts.get((index, name), shift)
            ax.annotate(f'${name}_{index}$', xy, xytext=shift, textcoords='offset points',
                        color=color, fontsize=11, va='center',
                        ha='left' if shift[0] >= 0 else 'right', zorder=16,
                        bbox=dict(fc='white', ec='none', pad=.4))
    if origin:
        xy = project(o)
        ax.scatter(*xy, s=22, marker='s', fc='white', ec=INK, lw=1, zorder=15)


def callout(ax, point, text, xytext, fontsize=10):
    ax.annotate(text, project(point), xytext=xytext, textcoords='offset points',
                fontsize=fontsize, color=INK, va='center',
                bbox=dict(fc='white', ec='none', pad=1.7),
                arrowprops=dict(arrowstyle='-', color='#80909d', lw=.8), zorder=20)


def skeleton(ax, links, width=8, labels=False):
    centers = np.array([links[f'xMateSR5_link{i}'][:3, 3]*1000 for i in range(1, 7)])
    j1, j2, j3, j4, j5, j6 = centers
    upper = j2+links['xMateSR5_link2'][:3, :3]@np.array([0, 0, 400.])
    fore = j3+links['xMateSR5_link3'][:3, :3]@np.array([0, 0, 400.])
    for i, points in [(0, [np.zeros(3), j1]), (2, [j2, upper, j3]),
                      (3, [j3, fore, j4]), (4, [j4, j5]), (5, [j5, j6])]:
        xy = project(points)
        ax.plot(xy[:, 0], xy[:, 1], color=LINKCOLORS[i], lw=width,
                solid_capstyle='round', alpha=.60, zorder=2)
    for point in centers:
        ax.scatter(*project(point), s=60, fc='#edf2f6', ec='#8b9ba6', lw=1.2, zorder=4)
    if labels:
        for i, point in enumerate(centers):
            if i == 1:
                continue
            ax.annotate('J1 / J2' if i == 0 else f'J{i+1}', project(point),
                        xytext=(-48, 7) if i < 3 else (10, -14), textcoords='offset points',
                        fontsize=9, color='#738591', zorder=9)
    return centers


def draw(robot, qdeg, filename, pose_name):
    q = np.deg2rad(qdeg)
    frames = dh_frames(q)
    _, _, links = robot.fk_jacobian(q)
    fig = plt.figure(figsize=(19, 13.5), facecolor='white')
    fig.text(.045, .952, 'SR5 · GÁN HỆ TRỤC DH TIÊU CHUẨN', fontsize=25, weight='bold')
    fig.text(.045, .918, pose_name+'  •  Hệ trục phải  •  Kích thước: mm', fontsize=12, color=ZCOLOR)
    fig.text(.96, .95, '$x$: đỏ    $z$: xanh\nyᵢ = zᵢ × xᵢ (lược trên hình)',
             ha='right', va='top', fontsize=11)
    fig.add_artist(plt.Line2D([.045,.96], [.89,.89], transform=fig.transFigure, color='#d4dfe7'))

    main = fig.add_axes([.035,.30,.53,.56])
    main.set_aspect('equal'); main.axis('off')
    centers = skeleton(main, links, labels=True)
    origins = np.array([F[:3,3]*1000 for F in frames])
    xy = project(origins)
    main.plot(xy[:,0],xy[:,1], ls=(0,(4,3)), color='#6c7d88', lw=1, zorder=3)
    for i, F in enumerate(frames):
        axes_at(main, F, i, 95 if i<2 else 65, labels=i<2)
    labels = [(-30,-14), (-70,22), (-65,-12), (30,25), (-55,28), (35,23), (35,-17)]
    for i,F in enumerate(frames):
        callout(main,F[:3,3]*1000,f'$O_{i}$',labels[i])
    O = project([0,0,0])
    main.add_patch(Polygon([O+[-38,-7],O+[38,-7],O+[55,-27],O+[-55,-27]],
                           fc='#e1e8ed',ec='#697d8b',zorder=1))
    # Draw the extended J4 axis through its DH origin O3.
    segment(main, origins[3]-70*frames[3][:3,2], origins[4]+60*frames[3][:3,2],
            color=ZCOLOR, lw=.9, ls=(0,(6,3)), zorder=1)
    allxy=project(np.vstack([origins, centers]))
    main.set_xlim(allxy[:,0].min()-200,allxy[:,0].max()+210)
    main.set_ylim(allxy[:,1].min()-80,allxy[:,1].max()+170)
    fig.text(.055,.285,'Đường màu: hình học khâu   •   ○: gốc khớp URDF   •   □: gốc hệ DH',fontsize=10)

    fig.text(.595,.847,'QUY TẮC GÁN TRỤC',fontsize=13,weight='bold')
    fig.text(.595,.813,'$z_{i-1}$ trùng trục quay dương của khớp $J_i$.\n'
             '$x_i$ theo pháp tuyến chung của $z_{i-1}$ và $z_i$.\n'
             '$O_i$ nằm trên $z_i$; không nhất thiết là tâm khớp.',
             fontsize=11,linespacing=1.8,va='top')
    fig.text(.595,.707,'$A_i=R_z(\\theta_i)\\,T_z(d_i)\\,T_x(a_i)\\,R_x(\\alpha_i)$',
             fontsize=15,color=ZCOLOR)

    # Close-ups retain the exact same projection and actual, unshifted origins.
    elbow=fig.add_axes([.60,.475,.17,.18])
    wrist=fig.add_axes([.80,.475,.17,.18])
    for ax, ids, title, length in [(elbow,[2,3],'PHÓNG TO KHUỶU',24),
                                   (wrist,[4,5,6],'PHÓNG TO CỔ TAY',32)]:
        ax.set_aspect('equal'); ax.set_facecolor('#f7fafc')
        for sp in ax.spines.values():sp.set_color('#d5e0e8')
        ax.set_xticks([]);ax.set_yticks([])
        pts=origins[ids]
        sk=project(pts)
        ax.plot(sk[:,0],sk[:,1],color='#a4b3bd',ls='--',lw=1)
        text_shifts={(2,'x'):(10,-1),(2,'z'):(-6,-12),(3,'x'):(-9,-2),(3,'z'):(7,8),
                     (4,'x'):(-7,-14),(4,'z'):(-9,-6),(5,'x'):(-12,8),(5,'z'):(9,2),
                     (6,'x'):(-10,-12),(6,'z'):(9,2)}
        for i in ids:
            axes_at(ax,frames[i],i,length,label_shifts=text_shifts)
            shifts={2:(-28,-5),3:(-28,12),4:(-28,3),5:(-23,17),6:(-23,-12)}
            callout(ax,origins[i],f'$O_{i}$',shifts[i],9)
        if ids==[2,3]:
            # Extended z3 leads to the physical J4 above/along the forearm.
            segment(ax,origins[3]-35*frames[3][:3,2],origins[3]+45*frames[3][:3,2],
                    color=ZCOLOR,lw=.8,ls=':',zorder=1)
        ax.set_xlim(sk[:,0].min()-55,sk[:,0].max()+65)
        ax.set_ylim(sk[:,1].min()-55,sk[:,1].max()+65)
        ax.set_title(title,fontsize=10,weight='bold',pad=10)

    fig.text(.595,.43,'ÁNH XẠ TRỤC KHỚP',fontsize=11,weight='bold')
    fig.text(.595,.402,'J1 ↔ z₀    J2 ↔ z₁    J3 ↔ z₂\nJ4 ↔ z₃    J5 ↔ z₄    J6 ↔ z₅',
             fontsize=11,linespacing=1.7,va='top')
    fig.text(.595,.335,'$z_6$ là trục hệ flange, không có khớp thứ 7.\n'
             '$O_3$ ở phần kéo dài trục J4; xem hình phóng to.',fontsize=10,linespacing=1.7,va='top')

    fig.text(.045,.245,'BẢNG DH TƯƠNG ỨNG VỚI CÁCH GÁN TRÊN',fontsize=12,weight='bold')
    ta=fig.add_axes([.045,.072,.915,.158]);ta.axis('off')
    theta=['$q_1$','$q_2-\\beta$','$q_3-\\beta$','$q_4$','$q_5$','$q_6$']
    rows=[[str(i+1),theta[i],['328','0','0','400','−136','103,5'][i],
           ['0','403,112887','−50','0','0','0'][i],
           ['−90°','180°','−90°','90°','−90°','0°'][i]] for i in range(6)]
    table=ta.table(cellText=rows,colLabels=['i','$\\theta_i$','$d_i$ (mm)','$a_i$ (mm)','$\\alpha_i$'],
                   cellLoc='center',loc='center',colWidths=[.08,.29,.21,.21,.21])
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.65)
    for (r,c),cell in table.get_celld().items():
        cell.set_facecolor('#e7f0f6' if r==0 else ('#f5f8fb' if r%2 else 'white'))
        cell.set_edgecolor('#d5e0e8');cell.set_linewidth(.65)
        if r==0:cell.set_text_props(weight='bold',color=ZCOLOR)
    fig.text(.045,.039,'β = atan2(400, 50) ≈ 82,874984°  •  q là góc URDF; θ là góc DH  •  a₃ và d₅ âm do chiều trục đã chọn.',fontsize=10)
    fig.text(.045,.017,'Một cách gán DH hợp lệ từ URDF, không phải bảng hãng công bố. Đã kiểm tra FK và các trục khớp; hệ {0} = base, hệ {6} = flange.',
             fontsize=9,color='#617888')
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{filename}.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    return {'q_degrees':list(qdeg),'frames_base_to_DH': [F.tolist() for F in frames]}


def draw_x2_alignment():
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    fig.text(.055,.94,'SR5 · TRỤC x₂ NẰM TRÊN ĐƯỜNG O₁O₂',fontsize=22,weight='bold')
    fig.text(.055,.89,r'$O_2-O_1=d_2z_1+a_2x_2=a_2x_2$   vì $d_2=0$',fontsize=17,color=ZCOLOR)
    checks=[]
    for box,qdeg,title in [([.06,.20,.40,.61],[0,0,0,0,0,0],'Cấu hình zero URDF'),
                            ([.56,.20,.40,.61],[0,30,-60,0,-90,0],'Tư thế gập')]:
        ax=fig.add_axes(box);ax.set_aspect('equal');ax.axis('off')
        F=dh_frames(np.deg2rad(qdeg))
        o1,o2=F[1][:3,3]*1000,F[2][:3,3]*1000
        x2,z1,z2=F[2][:3,0],F[1][:3,2],F[2][:3,2]
        residual=np.linalg.norm((o2-o1)-A[1]*1000*x2)
        assert residual<1e-9
        checks.append(float(residual))
        segment(ax,o1-35*x2,o2+145*x2,color='#b6c3ce',lw=1,ls='--',zorder=1)
        segment(ax,o1,o2,color='#466477',lw=5,zorder=2)
        arrow(ax,o2,o2+115*x2,XCOLOR,2.8)
        for o,z,name in [(o1,z1,'1'),(o2,z2,'2')]:
            segment(ax,o-70*z,o+90*z,color=ZCOLOR,lw=.8,ls=':',zorder=1)
            arrow(ax,o,o+90*z,ZCOLOR)
            tip=project(o+100*z)
            ax.text(*tip,f'$z_{name}$',color=ZCOLOR,fontsize=14,va='center',
                    ha='left' if project(z)[0]>=0 else 'right')
            ax.scatter(*project(o),s=65,fc='white',ec=INK,lw=1.3,zorder=15)
            ax.annotate(f'$O_{name}$',project(o),xytext=(-25,-15),textcoords='offset points',fontsize=14)
        tip=project(o2+125*x2)
        ax.text(*tip,'$x_2$',color=XCOLOR,fontsize=16,weight='bold',va='center')
        p1,p2=project([o1,o2]);v=p2-p1;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
        a,b=p1+60*n,p2+60*n
        ax.annotate('',b,a,arrowprops=dict(arrowstyle='<->',color='#77909f',lw=1.2))
        center=(a+b)/2+18*n
        ax.text(*center,'a₂ = 403,113 mm',color='#466477',fontsize=12,
                rotation=np.degrees(np.arctan2(v[1],v[0])),ha='center',va='bottom',rotation_mode='anchor')
        cloud=project([o1-100*z1,o1+110*z1,o2-100*z2,o2+110*z2,o2+155*x2])
        ax.set_xlim(cloud[:,0].min()-90,cloud[:,0].max()+70)
        ax.set_ylim(cloud[:,1].min()-65,cloud[:,1].max()+45)
        ax.set_title(title,fontsize=14,weight='bold',pad=15)
    fig.text(.055,.12,'Mũi tên x₂ đặt tại O₂, hướng từ O₁ qua O₂ và đi tiếp ra ngoài.\n'
             'Đường x₂ vuông góc cả z₁ và z₂; hai trục khớp này song song và ngược chiều.',fontsize=13,linespacing=1.8)
    fig.text(.055,.035,'Cùng dữ liệu và cách gán DH với hình trước; hình này lược các khớp khác để làm rõ x₂.',fontsize=10,color='#6d8494')
    for ext in ['png','svg']:
        fig.savefig(OUT/f'sr5_dh_x2_alignment.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    return checks


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    robot=Robot(ROOT.parent / 'trajectory/tee/generated/sr5_torch.urdf')
    checks=validate(robot)
    zero=draw(robot,[0,0,0,0,0,0],'sr5_standard_dh_zero','Cấu hình zero URDF: q₁ = … = q₆ = 0°')
    bent=draw(robot,[0,30,-60,0,-90,0],'sr5_standard_dh_bent','Tư thế gập: q = [0°, 30°, −60°, 0°, −90°, 0°]')
    draw_x2_alignment()
    report={'validation':checks,'beta_degrees':float(np.rad2deg(BETA)),
            'a_m':A.tolist(),'d_m':D.tolist(),'alpha_rad':ALPHA.tolist(),
            'theta_offset_rad':OFFSET.tolist(),'zero_configuration':zero,'bent_configuration':bent}
    (OUT/'sr5_standard_dh.json').write_text(json.dumps(report,indent=2))
    with (OUT/'sr5_standard_dh.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['joint','a_m','d_m','alpha_rad','theta_minus_q_rad'])
        w.writerows([[i+1,A[i],D[i],ALPHA[i],OFFSET[i]] for i in range(6)])
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
