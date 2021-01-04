FROM tensorflow/tensorflow:2.4.0-gpu
RUN pip install typer tqdm Pillow tf2onnx
ENV IMAGE_WEIGHTS="/code/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"