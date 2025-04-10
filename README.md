# rl_aliengo_ssrl

After cloning the repository:
```bash
git submodule update --init --recursive
pip install -e submodules/free-dog-sdk/
```

# Build docker
Inside dir ```rl_go1/docker``` run
```bash
docker build -t ros-jax-gpu .
```
# Start Docker
Inside dir: ```rl_go1```

```bash
xhost si:localuser:root
docker run --rm -it --ipc=host --gpus all --net=host -v .:/workspace --volume=$HOME/.Xauthority:/root/.Xauthority:rw -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged ros-jax-gpu bash
```

# After docker started init env
```bash
source /ssrl_entry.sh
```

# Build ros workspace
```bash
cd /workspace/ssrl_ws
catkin_make
```

# Use ros env
```bash
source devel/setup.sh
```

# SSRL Start to collect data
In separate terminals:
```bash
roscore
```
```bash
roslaunch ssrl_ros_go1 quadruped_comm.launch
```
```bash
roslaunch ssrl_ros_go1 quadruped_comm.launch
```
```bash
roslaunch ssrl_ros_go1 support_nodes_vicon.launch
```
```bash
rosrun ssrl_ros_go1 controller.py
```
- Generate data
    - Press the spacebar to start standing up the Aliengo;
    - When the Aliengo is standing, press the spacebar again to start walking;
    - The episode will terminiate automatically after 10 sec or if the robot falls;
    - To stop the episode early, press any key.
- Train
    - Run the following command in terminal to run training: `python ssrl_hardware/ssrl_ros_go1/scripts/train.py run_name=<RUN_NAME>`
        - Replace `<RUN_NAME>` with a descriptive name of the run. The name must not contain spaces or special characters except underscore.
- Repeat for the desired number of epochs.


# Potential problems

An error `liblcm.so.1: cannot open shared object file: No such file or directory` may occur after you run the command `roslaunch ssrl_ros_go1 support_nodes_vicon.launch`
To solve this problem run this command:
```bash
sudo ldconfig -v
```