"""
CIFAR10 Classification
Improving the classification accuracy by data augmentation
Tensorflow 2.0
"""
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##########################

NUM_TO_AUGMENT = 5
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128
NB_EPOCHS = 50  # changed
NB_CLASSES = 10

# load dataset
(X_train, y_train), (X_test, y_test) = \
    keras.datasets.cifar10.load_data()

print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape: ', X_test.shape)
print('y_test Shape: ', y_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# data augmentation
print('Augmenting training set images')
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# visualizing augmented data
# xtas, ytas = [], []
# for i in range(X_train.shape[0]):
#     num_aug = 0
#     x = X_train[i] # (32, 32, )
#     x = x.reshape((1, ) + x.shape) # (1, 32, 32, 3)
#     for x_aug in datagen.flow(x, batch_size=1,
#                               save_to_dir='preview',
#                               save_prefix='cifar',
#                               save_format='jpeg'):
#         if num_aug >= NUM_TO_AUGMENT:
#             break
#         xtas.append(x_aug[0])
#         num_aug += 1

# fit the dataset
datagen.fit(X_train)

# train

model = keras.models.Sequential([
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
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['accuracy'])
# train
history = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=10000, # X_train.shape[0],
        epochs=NB_EPOCHS)

# evaluate
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

print('Test Score 2: ', score[0])
print('Test Accuracy 2: ', score[1])

# save the model
model_json = model.to_json()
open('cifar10_arch.json', 'w').write(model_json)
# save weights
model.save_weights('cifar10_weights.h5', overwrite=True)
