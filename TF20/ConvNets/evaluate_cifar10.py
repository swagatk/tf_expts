"""
Evaluate A Trained Model on CIFAR10 dataset
Tensorflow 2.0
"""
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from skimage.io import imread

## Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
## ########################

# Load Keras Model
model_architecture = 'cifar10_arch.json'
model_weights = 'cifar10_weights.h5'
model = keras.models.model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# load images
cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
img_names = ['cat-standing.jpg', 'dog.jpg']
imgs = [np.transpose(resize(imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32')
        for img_name in img_names]
imgs = np.array(imgs) / 255.0

# train
optim = keras.optimizers.SGD()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optim, metrics=['accuracy'])

# predict
predictions = model.predict_classes(imgs)
print(predictions)
print('predicted labels: ', [cifar_labels[i] for i in predictions])
