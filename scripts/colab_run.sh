#!/usr/bin/env bash
# Run server to accept requests from google colab.
poetry run jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0