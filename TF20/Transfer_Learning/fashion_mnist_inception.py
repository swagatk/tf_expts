'''
Tensorflow 2.0 code
Transfer learning using inception-v3 model
dataset: fashion_mnist
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import numpy as np
from time import time

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
print('TF version: ', tf.__version__)

# Load the dataset
dataset, info = tfds.load("tf_flowers", with_info=True)
print(info)

dataset = dataset["train"]
total = 3670

train_set_size = total // 2
validation_set_size = total - train_set_size - train_set_size // 2
test_set_size = total - train_set_size - validation_set_size
print('train set size: ', train_set_size)
print('validation set size: ', validation_set_size)
print('test set size: ', test_set_size)

# create train, validation & test datasets
train, validation, test = (
    dataset.take(train_set_size),
    dataset.skip(train_set_size).take(validation_set_size),
    dataset.skip(train_set_size + validation_set_size).take(test_set_size)
)


def to_float_image(example):
    example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
    return example


def resize(example):
    example["image"] = tf.image.resize(example["image"], (299, 299))
    return example


# data pre-processing
train = train.map(to_float_image).map(resize)
validation = validation.map(to_float_image).map(resize)
test = test.map(to_float_image).map(resize)

# Load the Inception-V3 model from TFHub
num_classes = 5
model = tf.keras.Sequential([
    hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
        # output_shape=[2048],
        trainable=False
    ),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(num_classes)
])

# Training utilities
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(1e-3)
#global step variable to keep track of iterations
step = tf.Variable(1, name="global_step", trainable=False)

train_summary_writer = tf.summary.create_file_writer("./log/transfer/train")
validation_summary_writer = tf.summary.create_file_writer("./log/transfer/validation")

# Metrics
accuracy = tf.metrics.Accuracy()
mean_loss = tf.metrics.Mean(name="loss")


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss(labels, logits)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    step.assign_add(1)  # step += 1
    accuracy.update_state(labels, tf.argmax(logits, -1))
    return loss_value


# Configure the training
train = train.batch(32).prefetch(1)
validation = validation.batch(32).prefetch(1)
test = test.batch(32).prefetch(1)

num_epochs = 10
for epoch in range(num_epochs):
    start = time()
    for example in train:
        image, label = example["image"], example["label"]
        # print('size of image:', np.shape(image))
        loss_value = train_step(image, label)
        mean_loss.update_state(loss_value)
        if tf.equal(tf.math.mod(step, 10), 0):
            tf.print(
                step, "loss: ", mean_loss.result(), "accuracy: ", accuracy.result()
            )
            mean_loss.reset_states()
            accuracy.reset_states()
    end = time()
    print("Time per epoch: ", end - start)
    # Validation
    tf.print("## Validation - ", epoch)
    accuracy.reset_states()
    for example in validation:
        image, label = example["image"], example["label"]
        logits = model(image)
        accuracy.update_state(label, tf.argmax(logits, -1))
    tf.print("accuracy: ", accuracy.result())
    accuracy.reset_states()
