#!/bin/bash

cd /workspace
pip install -e submodules/ssrl/ssrl

cd /workspace/submodules
cp -a unitree_ros_to_real/ ../ssrl_aliengo_ws/src/
cp -a unitree_legged_sdk/ ../ssrl_aliengo_ws/src/unitree_ros_to_real/

cd /workspace/ssrl_aliengo_ws/src/unitree_ros_to_real/unitree_legged_sdk/

rm -rf build
mkdir build
cd build
cmake ..
make

source /opt/ros/noetic/setup.bash

roscd mocap_optitrack/
mkdir scripts
cp -a /workspace/mocap_files/vel_estimator_optitrack.py scripts/
chmod +x scripts/vel_estimator_optitrack.py
cp -a /workspace/mocap_files/mocap.yaml config/

cd /workspace/ssrl_aliengo_ws/
catkin_make
source devel/setup.bash
