# CLI to run an optimization to do neural style transfer
import os
import tarfile
import typer
import nst_jetson
import tensorflow as tf

from tensorflow.keras.utils import get_file

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg19/"
    "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
MAGENTA_MODEL = (
    "https://tfhub.dev/google/magenta/"
    "arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed"
)
MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(MODULE_PATH, "models")
DEFAULT_SAVED_MODEL = os.path.join(
    MODEL_PATH, "magenta_arbitrary-image-stylization-v1-256_2"
)


app = typer.Typer()


@app.command()
def setup():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    get_file(
        os.path.join(MODEL_PATH, "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"),
        WEIGHTS_PATH_NO_TOP,
        file_hash="253f8cb515780f3b799900260a226db6",
    )

    tar_file = get_file(
        os.path.join(MODEL_PATH, "magenta_arbitrary-image-stylization-v1-256_2.tar.gz"),
        MAGENTA_MODEL,
    )

    with tarfile.open(tar_file, "r:gz") as tar_handler:
        tar_handler.extractall(
            path=os.path.join(
                MODEL_PATH, "magenta_arbitrary-image-stylization-v1-256_2"
            )
        )


@app.command()
def apply_style(
    content_image_path: str,
    style_image_path: str,
    epochs=10,
    steps_per_epoch=100,
    style_weight=1e-2,
    content_weight=1e4,
    total_variation_weight=30,
):
    setup()
    # Notes: This is to avoid CUBLAS errors, if you know how to avoid this, please let me know.
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    content_image = nst_jetson.load_img(content_image_path)
    style_image = nst_jetson.load_img(style_image_path)
    nst_jetson.style_transfer(
        content_image=content_image,
        style_image=style_image,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        style_weight=style_weight,
        content_weight=content_weight,
        total_variation_weight=total_variation_weight,
    )


if __name__ == "__main__":
    app()