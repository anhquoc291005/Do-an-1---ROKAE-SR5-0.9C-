from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    root=Path(__file__).resolve().parent
    return LaunchDescription([
        Node(package='robot_state_publisher',executable='robot_state_publisher',
             parameters=[{'robot_description':(root/'generated/sr5_torch.urdf').read_text()}]),
        ExecuteProcess(cmd=['/usr/bin/python3',str(root/'replay.py')],output='screen'),
        Node(package='rviz2',executable='rviz2',arguments=['-d',str(root/'preview.rviz')])
    ])
