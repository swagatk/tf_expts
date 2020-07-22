'''
Tensorflow 2.0
Linear Regression Example
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print("Tensorflow version:", tf.__version__)

# actual weight = 2 and actual bias  = 0.9

x = np.linspace(0, 3, 120)
y = 2 * x + 0.9 + np.random.randn(*x.shape) * 0.3


# Linear Model
class LinearModel:
    def __call__(self, x):
        return self.Weight * x + self.Bias

    def __init__(self):
        self.Weight = tf.Variable(11.0)
        self.Bias = tf.Variable(12.0)


def loss(y, pred):
    return tf.reduce_mean(tf.square(y - pred))


# define optimizer
opt = tf.optimizers.SGD(learning_rate=0.1)


def train(linear_model, x, y):
    with tf.GradientTape() as t:
        trainable_variables = [linear_model.Weight,
                               linear_model.Bias]
        current_loss = loss(y, linear_model(x))
        gradients = t.gradient(current_loss,
                               trainable_variables)
        opt.apply_gradients(zip(gradients,
                                trainable_variables))


# Training
linear_model = LinearModel()
epochs = 80
for epoch_count in range(epochs):
    real_loss = loss(y, linear_model(x))
    train(linear_model, x, y)
    print(f"Epoch Count {epoch_count}: \
                    Loss value: {real_loss.numpy()}")

print('Estimated Weight: %.2f, Bias: %.2f' % (linear_model.Weight,
                                              linear_model.Bias))
# testing
yhat = linear_model(x)
plt.scatter(x, y, label='Data')
plt.plot(x, yhat.numpy(), 'r-', lw=2, label='estimate')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.grid()
plt.show()