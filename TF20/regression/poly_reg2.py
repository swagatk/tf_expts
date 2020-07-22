'''
TensorFlow 2.0 Code
Polynomial Regression
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print('TF Version:', tf.__version__)

# hyper-parameters
learning_rate = 0.1
training_epochs = 100

# Generate Input-Output Data
x = np.linspace(-1, 1, 101)
num_coeffs = 6
actual_coeffs = [1, 2, 3, 4, 5, 6]
y = 0
for i in range(num_coeffs):
    y += actual_coeffs[i] * np.power(x, i)
y += np.random.randn(*x.shape) * 1.5

# Define variables for training
w = tf.Variable([0.] * num_coeffs, trainable=True,
                dtype=tf.float32, name='parameters')

# define optimizer to use
opt = tf.optimizers.SGD(learning_rate)

# Model
def poly_model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = w[i] * tf.pow(tf.cast(X, tf.float32), i)
        terms.append(term)
    return tf.add_n(terms)


# Define Loss function
def loss(y, pred):
    return tf.reduce_mean(tf.square(y - pred))


# Training
for epoch in range(training_epochs):
    with tf.GradientTape() as t:
        trainable_variables = [w]
        y_pred = poly_model(x, w)
        current_loss = loss(y, y_pred)
        gradients = t.gradient(current_loss,
                               trainable_variables)
        opt.apply_gradients(zip(gradients,
                                trainable_variables))

    if epoch % 10 == 0:
        print("Epoch: %d, error = %.2f" \
                            % (epoch, current_loss))

np.set_printoptions(precision=3)
print('Estimated Polynomial coefficients:', w.numpy())

# Testing
yhat = poly_model(x, w)
plt.scatter(x, y, label='Data')
plt.plot(x, yhat, 'r-', label='Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.grid()
plt.show()



