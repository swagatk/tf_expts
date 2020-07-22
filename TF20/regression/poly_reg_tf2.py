"""
Tensorflow 2.0 code
polynomial regression
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('TF Version:', tf.__version__)

# hyper-parameters
learning_rate = 0.01
training_epochs = 40

trX = np.linspace(-1, 1, 101)

num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
trY += np.random.randn(*trX.shape) * 1.5


# placeholders and variables
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
w = tf.Variable([0.] * num_coeffs, dtype=tf.float32, name='parameters')

@tf.function
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = w[i] * tf.pow(tf.cast(X, tf.float32), i)
        #term = tf.scalar_mul(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

# optimizers

# loss function
for epoch in range(training_epochs):
    for (x, y) in zip(trX, trY):
        y_model = model(x, w)
        cost = tf.pow((y - y_model), 2)
        loss = lambda:
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        tf.keras.optimizers.SGD(learning_rate).minimize(cost, var_list=[w])


plt.scatter(trX, trY)
trY2 = 0
for i in range(num_coeffs):
    trY2 += w[i].numpy() * np.power(trX, i)
plt.plot(trX, trY2, 'r')
plt.show()

