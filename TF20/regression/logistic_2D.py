'''
Use Tensorflow 2.0
Logistic Regression as a binary classifier
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.1
training_epochs = 2000


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# Generate Synthetic Data
x1_label1 = np.random.normal(3, 1, 1000)
x2_label1 = np.random.normal(2, 1, 1000)
x1_label2 = np.random.normal(7, 1, 1000)
x2_label2 = np.random.normal(6, 1, 1000)

x1s = np.append(x1_label1, x1_label2)
x2s = np.append(x2_label1, x2_label2)
ys = np.asarray([0.] * len(x1_label1) + [1.] * len(x1_label2))

# Define variables & Optimizer
w = tf.Variable([0., 0., 0.], name="w", trainable=True)
# Optimizer to use
opt = tf.optimizers.SGD(learning_rate)


# model
def logistic_regression(x1, x2, w):
    return tf.sigmoid(w[2] * x2 + w[1] * x1 + w[0])


# cost function to minimize
def loss(pred, y):
    return tf.reduce_mean(-tf.math.log(pred) *
                          y - (1 - y) * tf.math.log(1 - pred))


# Training
prev_loss = 0
for epoch in range(training_epochs):
    with tf.GradientTape() as g:
        trainable_variables = [w]
        y_pred = logistic_regression(x1s, x2s, w)
        current_loss = loss(y_pred, ys)
        gradients = g.gradient(current_loss, trainable_variables)
        opt.apply_gradients(zip(gradients, trainable_variables))
        print(epoch, current_loss.numpy())
        if abs(prev_loss - current_loss) < 0.000001:
            break
        prev_loss = current_loss

# Testing & Plotting
x1_boundary, x2_boundary = [], []
for x1_test in np.linspace(0, 10, 100):
    for x2_test in np.linspace(0, 10, 100):
        z = sigmoid(-x2_test * w[2] - x1_test * w[1] - w[0])
        if abs(z - 0.5) < 0.01:
            x1_boundary.append(x1_test)
            x2_boundary.append(x2_test)

plt.scatter(x1_boundary, x2_boundary, c='b', marker='o', s=20,
                                    label='classifier boundary')
plt.scatter(x1_label1, x2_label1, c='r', marker='x', s=20,
                                    label='class1 data')
plt.scatter(x1_label2, x2_label2, c='g', marker='1', s=20,
                                    label='class2 data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.show()






