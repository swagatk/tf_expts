"""
MNIST Handwritten Digit Recognition using LeNet-CNN
Tensorflow 2.0
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##########################

# Hyper-parameters
NB_EPOCHS = 20
batch_size = 128
valid_split = 0.2
NB_CLASSES = 10
input_shape = (28, 28, 1)


# Load dataset
mnist_digit = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist_digit.load_data()

print('Training dataset shape: {}'.format(X_train.shape))
print('Shape of training Labels: {}'.format(y_train.shape))
print('Test dataset shape: {}'.format(X_test.shape))
print('Test Label shape: {}'.format(y_test.shape))

fig = plt.figure()
for i in range(16):
    plt.subplot(4, 4, 1+i)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')

print('Close the graph window to proceed ..')
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# create a LeNet CNN architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(20, kernel_size=5,
                        padding="SAME",
                        activation='relu',
                        input_shape=input_shape),
    keras.layers.MaxPool2D(pool_size=(2, 2),
                           strides=(2, 2)),
    keras.layers.Conv2D(50, kernel_size=5,
                        padding="SAME",
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2),
                           strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(NB_CLASSES,
                       activation='softmax')
])
model.summary()
keras.utils.plot_model(model, to_file='./lenet_model.png',
                       show_shapes=True, show_layer_names=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=NB_EPOCHS,
                    validation_split=valid_split)

score = model.evaluate(X_test, y_test)
print('Test score: ', score[0])
print('Test Accuracy: ', score[1])
print(history.history.keys())
# plot
fig, axes = plt.subplots(2)
plt.suptitle('Model Accuracy/Loss')
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_ylabel('Accuracy')
axes[0].legend(['train', 'test'], loc='best')

axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epochs')
print('Close the graph window to exit ....')
plt.show()



