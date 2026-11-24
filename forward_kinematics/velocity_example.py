"""Reproduce the Chapter 2 TCP velocity example using the RViz joint convention."""
import argparse
from pathlib import Path

import numpy as np

from fk import Robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=501)
    parser.add_argument("--demo", type=Path, default=Path(__file__).resolve().parents[1] / "trajectory/v_path")
    args = parser.parse_args()
    table = np.loadtxt(args.demo / "results/trajectory.csv", delimiter=",", skiprows=1, ndmin=2)
    k = args.sample
    if not 0 < k < len(table) - 1:
        parser.error("Select an interior sample for the central difference.")
    times = table[:, 0]
    if not times[k - 1] < times[k] < times[k + 1]:
        parser.error("The adjacent timestamps must be strictly increasing.")
    robot = Robot(args.demo / "generated/sr5_torch.urdf")
    q = table[:, 1:7]
    _, jacobian, _ = robot.fk_jacobian(q[k])
    dt = times[k + 1] - times[k - 1]
    twist = jacobian @ ((q[k + 1] - q[k - 1]) / dt)
    previous_pose = robot.fk_jacobian(q[k - 1])[0]
    next_pose = robot.fk_jacobian(q[k + 1])[0]
    fk_velocity = (next_pose[:3, 3] - previous_pose[:3, 3]) / dt
    print(f"Sample {k}, t = {times[k]:.6f} s")
    print("TCP velocity (mm/s):", twist[:3] * 1000)
    print("Angular velocity (rad/s):", twist[3:])
    print(f"Speed from Jacobian: {np.linalg.norm(twist[:3]) * 1000:.6f} mm/s")
    print(f"Speed from FK differences: {np.linalg.norm(fk_velocity) * 1000:.6f} mm/s")


if __name__ == "__main__":
    main()
