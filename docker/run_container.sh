#!/bin/bash

# Allow X server connections from root user
xhost +si:localuser:root

# Run Docker container
docker run --rm \
    -it \
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

