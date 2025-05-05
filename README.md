
# RL Robots

## PREREQUISITES
- Docker
- NVIDIA CUDA Toolkit (for GPU support)
- OptiTrack Motion Capture system
- Unitree AlienGo robot with SDK access

---

## QUICK START GUIDE

### 1. Clone Repository with Submodules
```bash
git clone -b ssrl_optitrack --recursive https://github.com/antsrt/rl_robots.git
cd rl_robots
```

### 2. Build Docker Image
```bash
cd docker
./build_docker.sh
```

### 3. Launch Container
```bash
./run_container.sh
```

### 4. Initialize Environment
```bash
source /ssrl_entry.sh
```
> This automatically handles ROS workspace setup and package builds

---

## OPTITRACK SETUP (MOTION CAPTURE)

### Configuration in Motive:
1. Calibrate capture volume
2. Create Rigid Body with Trackable ID = 1
3. Configure Data Streaming:
   - Broadcast Frame Data: Enabled
   - Local Interface: Not loopback (some ip-adress)
   - Up Axis: Z-Up
   - Transmission Type: Multicast

---

## SYSTEM ACTIVATION

### Terminal 1: Core Services
```bash
roscore
```

### Terminal 2: Robot Communication
```bash
roslaunch ssrl_ros_go1 quadruped_comm.launch
```

### Terminal 3: Support Nodes
```bash
roslaunch ssrl_ros_go1 support_nodes_vicon.launch
```

### Terminal 4: Motion Capture
```bash
roslaunch mocap_optitrack mocap.launch
```

### Terminal 5: Velocity Estimation
**REBOOT THIS NODE BEFORE EVERY DATA COLLECTION!!!**

```bash
rosrun mocap_optitrack vel_estimator_optitrack.py
```

### Terminal 6: Controller
```bash
rosrun ssrl_ros_go1 controller.py
```
> Select option 2 in rosrun menu

---

## DATA COLLECTION CONTROLS

| Key       | Action                      |
|-----------|-----------------------------|
| Space     | Start standing/walking      |
| Any key   | Emergency stop              |

> Episodes auto-terminate after 10s or on fall detection

---

## TRAINING PIPELINE

```bash
python ssrl_hardware/ssrl_ros_go1/scripts/train.py run_name=<RUN_NAME>
```
---

## TROUBLESHOOTING

### Missing LCM Library
**Error:** `liblcm.so.1: cannot open shared object file`
```bash
sudo ldconfig -v
```

### Motive Connection Issues
Check configuration at `rl_robots/mocap_files/mocap.yaml`:
- multicast_address
- command_port
- data_port

> Ensure these match Motive's streaming settings

---

## PROJECT STRUCTURE
```
rl_robots/
├── docker/               # Container configuration
├── ssrl_ros_go1/         # Robot control stack
├── mocap_optitrack/      # Motion capture integration
└── ssrl_hardware/        # RL training scripts