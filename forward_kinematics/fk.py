#!/usr/bin/python3
"""Independent URDF transform-product FK/Jacobian and numerical diagnostics."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def origin(element):
    T = np.eye(4)
    if element is not None:
        T[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return T


class Robot:
    def __init__(self, urdf):
        self.xml = E.parse(urdf).getroot()
        joints = {j.find('child').get('link'): j for j in self.xml.findall('joint')}
        self.chain = []
        link = 'welding_tcp'
        while link in joints:
            j = joints[link]
            self.chain.insert(0, j)
            link = j.find('parent').get('link')

    def fk_jacobian(self, q):
        T = np.eye(4)
        links = {'world': T.copy()}
        axes, centers = [], []
        for joint in self.chain:
            T = T @ origin(joint.find('origin'))
            if joint.get('type') == 'revolute':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                axis /= np.linalg.norm(axis)
                axes.append(T[:3, :3] @ axis)
                centers.append(T[:3, 3].copy())
                M = np.eye(4)
                M[:3, :3] = Rotation.from_rotvec(axis * q[len(axes)-1]).as_matrix()
                T = T @ M
            links[joint.find('child').get('link')] = T.copy()
        J = np.column_stack([np.r_[np.cross(a, T[:3, 3]-c), a] for a,c in zip(axes,centers)])
        return T, J, links


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SR5 forward kinematics and geometric Jacobian at the welding TCP.")
    parser.add_argument("angles", nargs=6, type=float, metavar="DEG")
    parser.add_argument("--urdf", type=Path, default=ROOT.parent / "trajectory/v_path/generated/sr5_torch.urdf")
    args = parser.parse_args()
    pose, jacobian, links = Robot(args.urdf).fk_jacobian(np.deg2rad(args.angles))
    np.set_printoptions(precision=6, suppress=True)
    print("Base to flange (translation in m):", links["xMateSR5_link6"], sep="\n")
    print("Base to TCP (translation in m):", pose, sep="\n")
    print("TCP geometric Jacobian (linear then angular):", jacobian, sep="\n")
