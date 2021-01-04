# Module that contains the optimization routine for the style transfer.
# Author: David Cardozo <germancho27@gmail.com>
import tensorflow as tf
from tqdm import tqdm
from .model import StyleContentModel
from .utils import clip_0_1, tensor_to_image

STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

CONTENT_LAYERS = ["block5_conv2"]


def style_transfer(
    content_image,
    style_image,
    epochs=10,
    steps_per_epoch=100,
    style_weight=1e-2,
    content_weight=1e4,
    total_variation_weight=30,
    style_layers=STYLE_LAYERS,
    content_layers=CONTENT_LAYERS,
):
    extractor = StyleContentModel(style_layers, content_layers)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    def style_content_loss(outputs):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]
        style_loss = tf.math.accumulate_n(
            [
                tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                for name in style_outputs.keys()
            ]
        )
        style_loss *= (style_weight / num_style_layers,)
        content_loss = tf.math.accumulate_n(
            [
                tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                for name in content_outputs.keys()
            ]
        )
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    image = tf.Variable(content_image)
    with tqdm(
        total=100,
        bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.3f}/{total_fmt} [{elapsed}<{remaining}",
    ) as pbar:

        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                train_step(image)
                pbar.update(100 / (epochs * steps_per_epoch))
    file_name = "stylized-image.png"
    tensor_to_image(image).save(file_name)