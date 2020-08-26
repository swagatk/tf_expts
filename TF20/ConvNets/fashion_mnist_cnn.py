"""
Fashion MNIST Dataset Classification
Tensorflow 2.0
ConvNets
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##########################

# load the dataset
mnist_fashion = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = \
                    mnist_fashion.load_data()

print('Training dataset shape: {}'.format(training_images.shape))
print('Shape of training Labels: {}'.format(training_labels.shape))
print('Test dataset shape: {}'.format(test_images.shape))
print('Test Label shape: {}'.format(test_labels.shape))

# scale the pixel values from [0, 255] to [0, 1]
training_images = training_images / 255.0
test_images = test_images / 255.0

# reshape the training /test data
training_images = training_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))


# build the model
cnn_model = keras.models.Sequential()
cnn_model.add(keras.layers.Conv2D(50, (3, 3), activation='relu',
                                  input_shape=(28, 28, 1), name='Conv2D_layer'))
cnn_model.add(keras.layers.MaxPool2D((2, 2), name='Maxpooling_2D'))
cnn_model.add(keras.layers.Flatten(name='Flatten'))
cnn_model.add(keras.layers.Dense(50, activation='relu', name='Hidden_Layer'))
cnn_model.add(keras.layers.Dense(10, activation='softmax', name='Output_layer'))
cnn_model.summary()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
# train
cnn_model.fit(training_images, training_labels, epochs=10)

# training evaluation
training_loss, training_accuracy = cnn_model.evaluate(training_images,
                                                      training_labels)
print('Training Accuracy {}'.format(round(float(training_accuracy), 2)))

# test evaluation
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)
print('Test Accuracy {}'.format(round(float(test_accuracy), 2)))

