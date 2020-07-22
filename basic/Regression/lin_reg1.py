'''
Linear Regression
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# user defined parameters
learning_rate = 0.01
training_epochs = 100

# input/output data
x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + \
          np.random.randn(*x_train.shape) * 0.33

# placeholders and variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable(0.0, name='Weights')


def model(x, w):
    return tf.multiply(x, w)


# predicted output
y_model = model(X, w)

cost = tf.square(Y - y_model)
train_op = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for (x, y) in zip(x_train, y_train):
            sess.run(train_op, feed_dict={X: x, Y: y})
    w_val = sess.run(w)

plt.scatter(x_train, y_train)
y_learned = x_train * w_val
plt.plot(x_train, y_learned, 'r-')
plt.show()






