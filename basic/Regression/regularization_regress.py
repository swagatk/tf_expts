'''
Tensorflow 1.x Code
Regularization in Regression
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# hyper-parameters
learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.

# function to splitting dataset into training & testing
def split_dataset(x_dataset, y_dataset, ratio):
    # This will work only for 1-D vector
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = int(ratio * x_dataset.size)
    x_train = x_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]]
    y_train = y_dataset[arr[0:num_train]]
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test


# Create dataset
x_dataset = np.linspace(-1, 1, 100)
num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0
for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

# train-test split
x_train, x_test, y_train, y_test = split_dataset(x_dataset, y_dataset, 0.7)

# create placeholders & variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable([0.] * num_coeffs, dtype=tf.float32, name='parameters')

# Define model, Cost, Optimizer
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(tf.cast(X, tf.float32), i))
        terms.append(term)
    return tf.add_n(terms)

y_model = model(X, w)
cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)),
                     tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))),
                     2*x_train.size)
train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# plot test data
plt.scatter(x_test, y_test, label='data')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for reg_lambda in np.linspace(0, 1, 5):
        # train
        for epoch in range(training_epochs):
            sess.run(train_opt, feed_dict={X: x_train, Y: y_train})

        # test
        yhat = model(np.sort(x_test), w)
        final_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
        print('reg lambda: %.2f, cost: %.5f' % (reg_lambda, final_cost))
        plt.plot(np.sort(x_test), yhat.eval(), '-',
                 label='$\lambda$={:.2f}'.format(reg_lambda))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    #plt.legend(bbox_to_anchor=(1.05, 1.0))
    plt.legend(loc='upper center')
    plt.show()


