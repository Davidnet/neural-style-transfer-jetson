#!/bin/env bash
#Shortcut to convert neural_style_transfer model to onnx
podman build -t davidnet/neural_style_transfer:gpu -f .
podman run --privileged -v "$(pwd)":/code:Z -w /code davidnet/neural_style_transfer:gpu python cli.py \
    setup
podman run --privileged -v "$(pwd)":/code:Z -w /code davidnet/neural_style_transfer:gpu python -m tf2onnx.convert \
    --opset 12 \
    --saved-model models/magenta_arbitrary-image-stylization-v1-256_2 \
    --output models/magenta_arbitrary-image-stylization-v1-256_2.onnx