#!/usr/bin/env bash
# Track of the command used to develop deepstream applications.
xhost +
podman run --privileged -it --rm \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY="$DISPLAY" \
	-w /opt/nvidia/deepstream/deepstream-5.0 \
	--net host \
	nvcr.io/nvidia/deepstream:5.0.1-20.09-triton bash