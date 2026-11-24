"""Display the vendor SR5 model at the pose used in Chapter 3."""
import math
import os
from pathlib import Path

import xacro
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    folder = Path(__file__).resolve().parent
    default_vendor = folder.parents[1] / "external" / "rokae_ros2"
    vendor = Path(os.getenv("ROKAE_ROS2_DIR", default_vendor)).resolve()
    description = vendor / "rokae_description"
    xml = xacro.process_file(str(description / "urdf/xMateSR5.urdf.xacro")).toxml()
    xml = xml.replace("package://rokae_description/", description.as_uri() + "/")
    zeros = {f"zeros.joint{i}": math.radians(degrees)
             for i, degrees in enumerate([0, 30, -60, 0, -90, 0], start=1)}
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": xml}]),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             parameters=[zeros]),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", str(folder / "robot_only.rviz")]),
    ])
