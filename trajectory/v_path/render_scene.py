#!/usr/bin/python3
"""Render the actual robot meshes and the planar V path, with a top view."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from analyze import Robot, ROOT, origin


def stl(path):
    b=Path(path).read_bytes()
    count=int.from_bytes(b[80:84],'little')
    dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    if len(b)!=84+50*count:raise ValueError('Expected binary STL')
    return np.frombuffer(b,dtype=dtype,offset=84,count=count)['v'].astype(float)


def transformed(vertices,T):
    return vertices @ T[:3,:3].T+T[:3,3]


def main():
    data=np.loadtxt(ROOT/'results/trajectory.csv',delimiter=',',skiprows=1)
    robot=Robot();_,_,links=robot.fk_jacobian(data[len(data)//2,1:])
    scene=json.loads((ROOT/'generated/scene.json').read_text())
    fig=plt.figure(figsize=(14,7),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    def draw(v,color):
        normals=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
        light=np.array([-.5,-.4,.8]);light/=np.linalg.norm(light)
        shade=.42+.58*np.abs(normals@light)
        colors=shade[:,None]*np.array(to_rgb(color))
        ax.add_collection3d(Poly3DCollection(v,facecolors=colors,edgecolors='none',alpha=1,rasterized=True))
    for link in robot.xml.findall('link'):
        name=link.get('name')
        if name not in links:continue
        visual=link.find('visual')
        if visual is None:continue
        mesh=visual.find('geometry/mesh')
        if mesh is None:continue
        vertices=stl(mesh.get('filename').removeprefix('file://'))
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        vertices=transformed(vertices*scale,links[name]@origin(visual.find('origin')))
        color='#bbc7d2' if name!='xMateSR5_base' else '#526675'
        draw(vertices,color)
    work=stl(ROOT/'generated/workpiece_mm.stl')*.001+np.array(scene['work_origin_m'])
    draw(work,'#4b7b91')
    p=np.array(scene['seam_endpoints_world_m']);ax.plot(*p.T,color='#ed4139',lw=3,label='Weld seam')
    p=np.array(scene['tcp_endpoints_world_m']);ax.plot(*p.T,color='#e8a91d',lw=2,label='TCP path (5 mm offset)')
    tcp=links['welding_tcp'];a=tcp[:3,3];b=a-.18*tcp[:3,2]
    ax.plot(*np.array([a,b]).T,color='#d88436',lw=7,label='Illustrative torch')
    ax.scatter(*a,color='#ffd529',s=45,depthshade=False)
    ax.set(xlim=(-.15,.8),ylim=(-.4,.4),zlim=(0,.9),xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)')
    ax.set_box_aspect((.95,.8,.9));ax.view_init(elev=24,azim=-135)
    ax.set_title('ROKAE SR5: V path A - B - C\nKinematic simulation at the corner',pad=20)
    ax.legend(loc='upper right')
    top=fig.add_subplot(122)
    from matplotlib.patches import Rectangle
    top.add_patch(Rectangle((370,-110),260,220,facecolor='#dce5eb',edgecolor='#607482'))
    seam=np.array(scene['seam_endpoints_world_m'])*1000
    top.plot(seam[:,0],seam[:,1],color='#ed4139',lw=5,label='V seam on plate')
    fk=np.loadtxt(ROOT/'results/fk.csv',delimiter=',',skiprows=1)
    top.plot(fk[:,2]*1000,fk[:,3]*1000,color='#14864e',lw=1.5,ls='--',label='TCP from FK')
    top.scatter(seam[:,0],seam[:,1],s=50,color='#233946',zorder=5)
    for label,p in zip('ABC',seam):
        top.annotate(label,(p[0],p[1]),xytext=(8,9),textcoords='offset points',fontsize=16,weight='bold')
    for i in range(2):
        a=.6*seam[i,:2]+.4*seam[i+1,:2];b=.4*seam[i,:2]+.6*seam[i+1,:2]
        top.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#152e48','lw':2})
    top.set(xlim=(350,650),ylim=(-130,130),xlabel='World X (mm)',ylabel='World Y (mm)',
            title=f'Top view: {scene["seam_length_m"]*1000:.1f} mm total\nStop 0.6 s at B; peak TCP speed 5 mm/s')
    top.set_aspect('equal');top.grid(alpha=.2);top.legend(loc='lower right')
    fig.savefig(ROOT/'results/scene.png',dpi=150)
    print(ROOT/'results/scene.png')


if __name__=='__main__':main()
