# Allow X server connections from root user
xhost si:localuser:root

# Run Docker container
docker run \
    --rm \                          # Remove container after exit
    -it \                          # Interactive terminal
    --ipc=host \                   # Use host's IPC namespace
    --gpus all \                   # Enable all GPUs
    --net=host \                   # Use host's network
    -v ..:/workspace \              # Mount parent directory to /workspace
    --volume=$HOME/.Xauthority:/root/.Xauthority:rw \  # X11 authentication
    -e NVIDIA_DRIVER_CAPABILITIES=all \                 # Enable all NVIDIA capabilities
    -e DISPLAY=$DISPLAY \          # Pass display variable
    -v /tmp/.X11-unix:/tmp/.X11-unix \                 # X11 socket
    --privileged \                 # Run with elevated privileges
    ros-jax-gpu \                 # Container image name
    bash -c "cd workspace/ssrl_ws && source devel/setup.bash && bash"  # Command to run with initial setup