'''
Tensorflow 1.0  Code
Linear Regression
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print('TF Version:', tf.__version__)

# hyper-parameters
learning_rate = 0.01
training_epochs = 40

# Create Input-Output dataset
trX = np.linspace(-1, 1, 101)
num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
trY += np.random.randn(*trX.shape) * 1.5

# placeholders and variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable([0.] * num_coeffs, name="parameters", dtype=tf.float32)

def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

# model output
y_model = model(X, w)

# loss function
cost = tf.pow((Y - y_model), 2)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    # get weights after training
    w_val = sess.run(w)
    print(w_val)

plt.scatter(trX, trY)
trY2 = 0
for i in range(num_coeffs):
    trY2 += w_val[i] * np.power(trX, i)
plt.plot(trX, trY2, 'r')
plt.show()



