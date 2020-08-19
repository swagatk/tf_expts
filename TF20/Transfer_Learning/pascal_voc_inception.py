"""
Tensorflow 2.0
Transfer learning using Inception-V3
Dataset: Pascal VOC 2007 dataset
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_hub as hub

### Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
###
# load the dataset
(train, test, validation), info = tfds.load(
    "voc2007", split=["train", "test", "validation"], with_info=True
)

# display few images
# with tf.device("/CPU:0"):
#     for row in train.take(5):
#         obj = row["objects"]
#         image = tf.image.convert_image_dtype(row["image"], tf.float32)
#
#         for idx in tf.range(tf.shape(obj["label"])[0]):
#             image = tf.squeeze(
#                 tf.image.draw_bounding_boxes(
#                     images=tf.expand_dims(image, axis=[0]),
#                     boxes=tf.reshape(obj["bbox"][idx], (1, 1, 4)),
#                     colors=tf.reshape(tf.constant((1.0, 1.0, 0, 0)), (1, 4))
#                 ),
#                 axis=[0]
#             )
#             print(
#                 "labels: ", info.features["objects"]["label"].int2str(obj["label"][idx])
#             )
#             plt.imshow(image)
#             plt.show()
#
####
# apply filter to create a dataset of images with single object annotated
def filter(dataset):
    return dataset.filter(lambda row: tf.equal(tf.shape(row["objects"]["label"])[0], 1))

train, test, validation = filter(train), filter(test), filter(validation)

# Load Inception model from TF Hub
inputs = tf.keras.layers.Input(shape=(299, 299, 3))
net = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
    output_shape=[2048],
    trainable=False
)(inputs)
net = tf.keras.layers.Dense(512)(net)
net = tf.keras.layers.ReLU()(net)
coordinates = tf.keras.layers.Dense(4, use_bias=False)(net)
regressor = tf.keras.Model(inputs=inputs, outputs=coordinates)

# prepare the dataset
def prepare(dataset):
    def _fn(row):
        row["image"] = tf.image.convert_image_dtype(row["image"], tf.float32)
        row["image"] = tf.image.resize(row["image"], (299, 299))
        return row
    return dataset.map(_fn)

# convert pixel values to [0, 1] and resize the images
train, test, validation = prepare(train), prepare(test), prepare(validation)

# Loss function
# L2 Loss
def l2(y_true, y_pred):
    return tf.reduce_mean(
        tf.square(y_pred - tf.squeeze(y_true, axis=[1]))
    )

# Compute IOU
def iou(pred_box, gt_box, h, w):

    # swap absolute coordinates to pixel coordinates
    # (y_min, x_min, y_max, x_max) -> (x_min, y_min, x_max, y_max)
    def _swap(box):
        return tf.stack([box[1] * w, box[0] * h, box[3] * w, box[2] * h])
    pred_box = _swap(pred_box)
    gt_box = _swap(gt_box)

    box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    xx1 = tf.maximum(pred_box[0], gt_box[0])
    yy1 = tf.maximum(pred_box[1], gt_box[1])
    xx2 = tf.minimum(pred_box[2], gt_box[2])
    yy2 = tf.minimum(pred_box[3], gt_box[3])
    w = tf.maximum(0, xx2 - xx1)
    h = tf.maximum(0, yy2 - yy1)
    inter = w * h
    return inter / (box_area + gt_area - inter)

# IOU Threshold
iou_threshold = 0.75
precision_metric = tf.metrics.Precision()
# draw function takes a dataset, the model and
# the current step and uses them to draw both
# the ground truth and the predicted boxes
def draw(dataset, regressor, step):
    with tf.device("/CPU:0"):
        row = next(iter(dataset.take(3).batch(3)))
        images = row["image"]
        obj = row["objects"]
        boxes = regressor(images)
        tf.print(boxes)
        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(boxes, (-1, 1, 4)),
            colors=tf.reshape(tf.constant((1.0, 0.0, 0, 0)), (1, 4))
        )
        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(obj["bbox"], (-1, 1, 4)),
            colors=tf.reshape(tf.constant((0.0, 1.0, 0, 0)), (1, 4))
        )
        tf.summary.image("images", images, step=step)

        # precision
        true_labels, predicted_labels = [], []
        for idx, predicted_box in enumerate(boxes):
            iou_value = iou(predicted_box, tf.squeeze(obj["bbox"][idx]), 299, 299)
            true_labels.append(1)
            predicted_labels.append(1 if iou_value >= iou_threshold else 0)

        precision_metric.update_state(true_labels, predicted_labels)
        tf.summary.scalar("precision", precision_metric.result(), step=step)


# training configuration
optimizer = tf.optimizers.Adam()
epochs = 100
batch_size = 32
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

# Define writers for tensorboard
train_writer, validation_writer =(
    tf.summary.create_file_writer("log/train"),
    tf.summary.create_file_writer("log/validation")
)

# training step
@tf.function
def train_step(image, coordinates):
    with tf.GradientTape() as tape:
        loss = l2(coordinates, regressor(image))
    gradients = tape.gradient(loss, regressor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, regressor.trainable_variables))
    return loss

# Training
train_batches = train.cache().batch(batch_size).prefetch(1)
with train_writer.as_default():
    for _ in tf.range(epochs):
        for batch in train_batches:
            obj = batch["objects"]
            coordinates = obj["bbox"]
            loss = train_step(batch["image"], coordinates)
            tf.summary.scalar("loss", loss, step=global_step)
            global_step.assign_add(1)
            if tf.equal(tf.math.mod(global_step, 10), 0):
                tf.print("step: ", global_step, " loss: ", loss)
                with validation_writer.as_default():
                    draw(validation, regressor, global_step)
                with train_writer.as_default():
                    draw(train, regressor, global_step)