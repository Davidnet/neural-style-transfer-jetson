#!/bin/env bash
#Shortcut to run neural_style_transfer
podman build -t davidnet/neural_style_transfer:gpu -f .
podman run --privileged -v "$(pwd)":/code:Z -w /code davidnet/neural_style_transfer:gpu python cli.py \
    setup
# podman run --privileged -v "$(pwd)":/code:Z -w /code davidnet/neural_style_transfer:gpu python cli.py \
#