'''
We apply a CNN to classify CIFAR10 data
- We have 2 conv layers
- Provides test accuracy of about 68%
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

# Apply 32 different 5x5 filters along 3 channels
conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)
# output is 16x16x32

# Apply 64 5x5 filters along 32 channels
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
# output is 8x8x64
conv2_flat = tf.reshape(conv2_pool, [-1, 8*8*64])
# Fully-connected layer
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = full_layer(full1_drop, 10)
print("shape of output: {}".format(tf.shape(y_conv)))

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


STEPS = 3000
BATCH_SIZE = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Training ....')
    for i in range(STEPS):
        batch = cifar.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 0.5
        })
        acc = sess.run(accuracy, feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 0.5
        })
        if i % 100 == 0:
            print("Epochs: {}/{}, Accuracy: {:.4}".format(i, STEPS, acc))

    # testing phase
    print('Testing ...')
    test(sess)

