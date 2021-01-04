# Module to agroup utils for neural style transfer
# Author: David Cardozo <germancho27@gmail.com>
from typing import List
import tensorflow as tf
import numpy as np
import PIL.Image
import os

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_FILENAME = "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"


def vgg_layers(layer_names: List[str]) -> tf.keras.Model:
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Use a pretrained VGG19 model.
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights=os.environ.get(
            "IMAGE_WEIGHTS", os.path.join(MODULE_PATH, "..", "models", MODEL_FILENAME)
        ),
    )
    vgg.trainable = False

    output = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], output)
    return model


def gram_matrix(input_tensor):
    """Implementation of the Gram calculation using:
    $$
    G_{cd}^l = \frac{\sum_{ij}F_{ijc}^l(x)F_{ijd}^l(x)}{IJ}
    $$
    """
    # Indices represent batches, and the rest of indeces (ijc, ijd)
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    # Assuming all the elements of the batch have the same shape dimensions
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = np.array(tensor * 255, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        # Assert we are only looking at one image
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img