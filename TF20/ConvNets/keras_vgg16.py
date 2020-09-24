"""
Using Keras VGG16 to recognize a cat image
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import cv2

### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##########################

# Prebuilt VGG model with pre-trained weights on imagenet
model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

img = image.load_img('./test_images/steam-locomotive.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)  # This is important to include this step.

# im = cv2.resize(cv2.imread('./test_images/steam-locomotive.jpeg'), (224, 224))
# im = np.expand_dims(im, axis=0)

# predict
# out = model.predict(im)  # this give wrong prediction
out = model.predict(x)  # this gives correct prediction
plt.plot(out.ravel())
plt.show()
print(np.argmax(out)) # 820 for steaming train
print('Predicted: ', decode_predictions(out, top=3)[0])