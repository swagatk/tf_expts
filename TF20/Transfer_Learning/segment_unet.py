"""
Tensorflow 2.0
Semantic Segmentation using a U-NET Architecture
Incomplete code
"""
import tensorflow as tf
import math
from tensorflow.keras.utils import plot_model

def downsample(depth):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                depth, 3, strides=2, padding="same",
                kernel_initializer="he_normal"
            ),
            tf.keras.layers.LeakyReLU(),
        ]
    )

def upsample(depth):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(
                depth, 3, strides=2, padding="same",
                kernel_initializer="he_normal"
            ),
            tf.keras.layers.ReLU(),
        ]
    )

def get_unet(input_size=(256, 256, 3), num_classes=21):
    # Downsample from 256x256 to 4x4 while adding depth
    # using powers of 2, starting from 2**5. cap to  512
    encoders = []
    for i in range(2, int(math.log2(256))):
        depth = 2 ** (i + 5)
        if depth > 512:
            depth = 512
        encoders.append(downsample(depth=depth))

    # Upsample from 4x4 to 256x256 reducing depth
    decoders = []
    for i in reversed(range(2, int(math.log2(256)))):
        depth = 2 ** (i + 5)
        if depth < 32:
            depth = 32
        if depth > 512:
            depth = 512
        decoders.append(upsample(depth=depth))

    # Build model by invoking encoder layers with correct input
    inputs = tf.keras.layers.Input(input_size)
    concat = tf.keras.layers.Concatenate()
    x = inputs
    #Encoders - downsample loop
    skips = []
    for conv in encoders:
        x = conv(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    #Decoder: input + skip connection
    for deconv, skip in zip(decoders, skips):
        x = deconv(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Add the last layer on the top and define the model
    last = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2, padding="same",
        kernel_initializer="he_normal" )
    outputs = last(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = get_unet()
plot_model(model, to_file="unet.png")