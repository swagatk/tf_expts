"""
Tensorflow 2.0
Double headed Network
Pascal VOC 2007 dataset
Task - Simultaneous localization and classification
pre-trained Inception-V3 network is used as feature extractor
"""
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_datasets as tfds


### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
###
############################################
# Dataset preparation

# load the dataset
(train, test, validation), info = tfds.load(
    "voc2007", split=["train", "test", "validation"], with_info=True
)

# apply filter to create a dataset of images with single object annotated
def filter(dataset):
    return dataset.filter(lambda row: tf.equal(tf.shape(row["objects"]["label"])[0], 1))

# prepare the dataset - convert pixel values to [0, 1] and resize the images
def prepare(dataset):
    def _fn(row):
        row["image"] = tf.image.convert_image_dtype(row["image"], tf.float32)
        row["image"] = tf.image.resize(row["image"], (299, 299))
        return row
    return dataset.map(_fn)


# apply filter to create a dataset of single objects
train, test, validation = filter(train), filter(test), filter(validation)

# convert pixel values to [0, 1] and resize the images
train, test, validation = prepare(train), prepare(test), prepare(validation)

############################################
# Defining a Model
# Define a double-headed network model
num_classes = 20  # categories of objects in images
inputs = tf.keras.layers.Input(shape=(299, 299, 3))
net = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
    output_shape=[2048],
    trainable=False
)(inputs)
regression_head = tf.keras.layers.Dense(512, activation='relu')(net)
coordinates = tf.keras.layers.Dense(4, use_bias=False)(regression_head)
classification_head = tf.keras.layers.Dense(1024, activation='relu')(net)
classification_head = tf.keras.layers.Dense(128, activation='relu')(classification_head)
classification_head = tf.keras.layers.Dense(num_classes, use_bias=False)(classification_head)

model = tf.keras.Model(inputs=inputs, outputs=[coordinates, classification_head])
##########################################
# Functions needed for Training

# classification Loss Functions
clfn_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# regression loss function
def l2(y_true, y_pred):
    return tf.reduce_mean(
        tf.square(y_pred - tf.squeeze(y_true, axis=[1]))
    )


# Compute IOU measure
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
    row = next(iter(dataset.take(3).batch(3)))
    images = row["image"]
    obj = row["objects"]
    boxes = model(images)[0]
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


# training step
@tf.function
def train_step(image, coordinates, labels):
    lambda1, lambda2 = 0.5, 0.5
    with tf.GradientTape() as tape:
        prediction = model(image)
        regression_loss = l2(coordinates, prediction[0])
        classification_loss = clfn_loss(labels, prediction[1])
        loss = lambda1 * classification_loss + lambda2 * regression_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
############################################
# training configuration
optimizer = tf.optimizers.Adam(1e-3)
epochs = 100
batch_size = 32
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
train_writer, validation_writer =(
    tf.summary.create_file_writer("log/train"),
    tf.summary.create_file_writer("log/validation")
)


# Training
train_batches = train.cache().batch(batch_size).prefetch(1)
with train_writer.as_default():
    for _ in tf.range(epochs):
        for batch in train_batches:
            obj = batch["objects"]
            coordinates = obj["bbox"]
            category_labels = obj["label"]
            loss = train_step(batch["image"], coordinates, category_labels)
            tf.summary.scalar("loss", loss, step=global_step)
            global_step.assign_add(1)
            if tf.equal(tf.math.mod(global_step, 10), 0):
                tf.print("step: ", global_step, " loss: ", loss)
                with validation_writer.as_default():
                    draw(validation, model, global_step)
                with train_writer.as_default():
                    draw(train, model, global_step)
