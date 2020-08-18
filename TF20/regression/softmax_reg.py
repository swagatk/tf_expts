'''
Tensorflow 2.0 code
understanding softmax regression
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##############################
# prepare the dataset
x1_label0 = np.random.normal(1, 1, (100, 1))
x2_label0 = np.random.normal(1, 1, (100, 1))
x1_label1 = np.random.normal(5, 1, (100, 1))
x2_label1 = np.random.normal(4, 1, (100, 1))
x1_label2 = np.random.normal(8, 1, (100, 1))
x2_label2 = np.random.normal(0, 1, (100, 1))

plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='_', s=60)
plt.show()

# combines all data into a one big matrix
xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
xs = np.vstack((xs_label0, xs_label1, xs_label2))

labels = np.array([[1., 0., 0.]] * len(x1_label0) +
                   [[0., 1., 0.]] * len(x1_label1) +
                   [[0., 0., 1.]] * len(x1_label2))

# shuffle the dataset
arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
xs = xs[arr, :]
labels = labels[arr, :]

# test datasets
test_x1_label0 = np.random.normal(1, 1, (10, 1))
test_x2_label0 = np.random.normal(1, 1, (10, 1))
test_x1_label1 = np.random.normal(5, 1, (10, 1))
test_x2_label1 = np.random.normal(4, 1, (10, 1))
test_x1_label2 = np.random.normal(8, 1, (10, 1))
test_x2_label2 = np.random.normal(0, 1, (10, 1))
test_xs_label0 = np.hstack((test_x1_label0,
                            test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1,
                            test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2,
                            test_x2_label2))
test_xs = np.vstack((test_xs_label0,
                     test_xs_label1,
                     test_xs_label2))
test_labels = np.array([[1., 0., 0.]] * 10 +
                       [[0., 1., 0.]] * 10 +
                       [[0., 0., 1.]] * 10)

train_size, num_features = xs.shape

######################
# define hyperparameters
learning_rate = 0.001
training_epochs = 1000
num_labels = 3
batch_size = 100


# Model
class SoftmaxModel:
    def __call__(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.nn.softmax(tf.matmul(x, self.Weight) + self.Bias)

    def __init__(self):
        self.Weight = tf.Variable(tf.zeros([num_features, num_labels]),
                                  trainable=True, dtype=tf.float32)
        self.Bias = tf.Variable(tf.zeros([num_labels]),
                                trainable=True, dtype=tf.float32)


def loss(pred, y):
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return -tf.reduce_sum(y * tf.math.log(pred))


def accuracy(pred, y):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, "float"))


# optimizer
opt = tf.optimizers.SGD(learning_rate)

# Create a Model and Train
model = SoftmaxModel()
for step in range(training_epochs * train_size // batch_size):
    offset = (step * batch_size) % train_size

    batch_xs = xs[offset:(offset + batch_size), :]
    batch_labels = labels[offset:(offset + batch_size)]

    with tf.GradientTape() as tape:
        training_variables = [model.Weight, model.Bias]
        pred = model(batch_xs)
        current_loss = loss(pred, batch_labels)
        gradients = tape.gradient(current_loss, training_variables)
        opt.apply_gradients(zip(gradients, training_variables))
    print(step, current_loss.numpy())

# final values of weights and bias
W = model.Weight.numpy()
b = model.Bias.numpy()
print('Weight: ', W)
print('Bias: ', b)

# Testing
print('accuracy: ', accuracy(model(test_xs), test_labels).numpy())

#
# def sigmoid(x):
#     return 1. / (1. + np.exp(-x))
#
#
# # plotting
# x1_boundary1, x2_boundary1 = [], []
# x1_boundary2, x2_boundary2 = [], []
# x1_boundary3, x2_boundary3 = [], []
# for x1_test in np.linspace(-2, 10, 1000):
#     for x2_test in np.linspace(-2, 10, 1000):
#         z1 = sigmoid(-W[0][0] * x1_test - W[1][0] * x2_test - b[0])
#         z2 = sigmoid(-W[0][1] * x1_test - W[1][1] * x2_test - b[1])
#         z3 = sigmoid(-W[0][2] * x1_test - W[1][2] * x2_test - b[2])
#
#         if abs(z1 - 0.5) < 0.01:
#             x1_boundary1.append(x1_test)
#             x2_boundary1.append(x2_test)
#         elif abs(z2 - 0.5) < 0.01:
#             x1_boundary2.append(x1_test)
#             x2_boundary2.append(x2_test)
#         elif abs(z3 - 0.5) < 0.01:
#             x1_boundary3.append(x1_test)
#             x2_boundary3.append(x2_test)
#
#
# plt.scatter(x1_boundary1, x2_boundary1, c='m', marker='o', s=20)
# plt.scatter(x1_boundary2, x2_boundary2, c='k', marker='x', s=20)
# plt.scatter(x1_boundary3, x2_boundary3, c='c', marker='+', s=20)
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.show()
#



