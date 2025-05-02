#!/bin/bash

# Allow X server connections from root user
xhost +si:localuser:root

# Check if the container is already running
container_id=$(docker ps -aqf "name=ssrl-ros-gpu")
if [ -z "$container_id" ]; then
    # Run Docker container
    docker run --rm \
        -it \
        --name ssrl-ros-gpu \
        --ipc=host \
        --gpus all \
        --net=host \
        -v "$(pwd)/..:/workspace" \
        --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --privileged \
        ssrl-ros-gpu \
        bash -c "source /opt/ros/noetic/setup.bash && source /workspace/ssrl_aliengo_ws/devel/setup.bash && bash"
else
    # Execute command in the already running container
    docker exec -it $container_id bash -c "source /opt/ros/noetic/setup.bash && source /workspace/ssrl_aliengo_ws/devel/setup.bash && bash"
fi