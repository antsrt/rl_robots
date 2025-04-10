# rl_robots

After cloning the repository:
```bash
git submodule update --init --recursive
```

# Build docker
Inside dir ```rl_robots/docker``` run
```bash
./build_docker.sh
```
# Start Docker
Inside dir: ```docker```

```bash
./run_container.sh
```

# After docker started init env
```bash
source /ssrl_entry.sh
```
It automatically build all packages, create ROS env, build and source it.

# SSRL Start to collect data
In separate terminals:
```bash
roscore
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