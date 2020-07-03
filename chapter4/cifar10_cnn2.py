'''
More clean up model
- We add 3 Conv layers between pooling layers
- Provides accuracy of about 76% for 20K training steps
'''

import tensorflow as tf
import numpy as np
from mnist2 import conv_layer, max_pool_2x2, full_layer
from cifar10 import CifarDataManager



# Get the dataset
cifar = CifarDataManager()
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

C1, C2, C3 = 30, 50, 80
F1 = 500

## CNN Model
# layer1: Apply 32 different 5x5 filters along 3 channels
conv1_1 = conv_layer(x, shape=[3, 3, 3, C1])
conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])
conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])
conv1_pool = max_pool_2x2(conv1_3)
conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])
conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])
conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])
conv2_pool = max_pool_2x2(conv2_3)
conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])
conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])
conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])
conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                            padding='SAME')
conv3_flat = tf.reshape(conv3_pool, [-1, C3])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full1 = tf.nn.relu(full_layer(conv3_flat, F1))
full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)
y_conv = full_layer(full1_drop, 10)
# ------------------------------


# Define loss functions
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={
        x: X[i],
        y_: Y[i],
        keep_prob: 1.0}) for i in range(10)])
    print("Accuracy: {:.4}".format(acc * 100))


STEPS = 20000
BATCH_SIZE = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Training ....')
    for i in range(STEPS):
        batch = cifar.train.next_batch(BATCH_SIZE)
        _, acc = sess.run([train_step, accuracy], feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 0.5
        })
        # acc = sess.run(accuracy, feed_dict={
        #     x: batch[0],
        #     y_: batch[1],
        #     keep_prob: 0.5
        # })
        if i % 100 == 0:
            print("Epochs: {}/{}, Accuracy: {:.4}".format(i, STEPS, acc))

    # testing phase
    print('Testing ...')
    test(sess)

