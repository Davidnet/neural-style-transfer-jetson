#!/bin/env bash
# Start an accelerated container with using google Colab.
# Author: David Cardozo <germancho27@gmail.com>
podman run --rm --privileged -p 8888:8888 tensorflow:2.4.0-gpu-jupyter bash -c \
    "jupyter serverextension enable --py jupyter_http_over_ws && \
    jupyter notebook --notebook-dir=/tf \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --ip localhost --no-browser --allow-root"