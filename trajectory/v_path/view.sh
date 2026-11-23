#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=90
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
exec /usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch ./preview.launch.py
