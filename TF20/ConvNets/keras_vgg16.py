"""
Using Keras VGG16 to recognize a cat image
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Prebuilt VGG model with pre-trained weights on imagenet
model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
im = cv2.resize(cv2.imread('./test_images/steam-locomotive.jpeg'), (224, 224))
im = np.expand_dims(im, axis=0)

# predict
out = model.predict(im)
plt.plot(out.ravel())
plt.show()
print(np.argmax(out)) # 820 for steaming train