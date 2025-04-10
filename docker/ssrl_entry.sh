#!/bin/bash

cd /workspace
pip install -e submodules/ssrl/ssrl

cd lcm/

rm -rf build

mkdir build
cd build
cmake ..
make 
sudo make install
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

cd /workspace/ssrl_ws/src/unitree_ros_to_real/unitree_legged_sdk/

rm -rf build
mkdir build
cd build
cmake ..
make

cd /workspace/ssrl_ws/
catkin_make
source devel/setup.bash


