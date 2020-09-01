"""
CIFAR10 Dataset Classification
Tensorflow 2.0

This uses a deeper network to provide a test accuracy of about 72%
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##########################

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128
NB_EPOCHS = 40  # changed
NB_CLASSES = 10
VALID_SPLIT = 0.2

# load dataset
(X_train, y_train), (X_test, y_test) = \
    keras.datasets.cifar10.load_data()

print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape: ', X_test.shape)
print('y_test Shape: ', y_test.shape)

# fig = plt.figure()
# plt.suptitle('Sample Images')
# for i in range(9):
#     plt.subplot(3, 3, 1+i)
#     plt.imshow(X_train[i, :, :])
#     plt.axis('off')
#
# print('Close the graph window to proceed ...')
# plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
########

model2 = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='SAME',
                        activation='relu',
                        input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)),
    keras.layers.Conv2D(32, (3, 3), padding='SAME',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), padding='SAME',
                        activation='relu'),
    keras.layers.Conv2D(64, 3, 3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(NB_CLASSES, activation='softmax')
])
model2.summary()

model2.compile(loss='sparse_categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['accuracy'])
# train
history = model2.fit(X_train, y_train, batch_size=BATCH_SIZE,
                     epochs=NB_EPOCHS,
                     validation_split=VALID_SPLIT)
# evaluate
score = model2.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('Test Score 2: ', score[0])
print('Test Accuracy 2: ', score[1])
