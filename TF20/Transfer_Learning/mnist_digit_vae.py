"""
Variational Autoencoders
Tensorflow 2.0
MNIST handwritten character
"""
import time
import numpy as np
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# ###

# load dataset
(training_data, _), (test_data, _) = mnist.load_data()
training_data = training_data.reshape(training_data.shape[0], 28, 28, 1).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')

# Analyze the data
print('Shape of training data: {}'.format(np.shape(training_data)))
print('Shape of test data: {}'.format(np.shape(test_data)))

fig = plt.figure(figsize=(4, 4))
for i in range(16):
  idx = np.random.choice(np.shape(training_data)[0])
  plt.subplot(4, 4, i+1)
  plt.imshow(training_data[idx, :,:,0], cmap='gray')
  plt.axis('off')

print('Close the figure window to proceed ..')
plt.show()
############
## Data pre-processing
# normalization
training_data = training_data / 255.0
test_data = test_data / 255.0

# binarization: convert pixel values to 0/1
training_data[training_data >= 0.5] = 1.0
training_data[training_data < 0.5] = 0.0
test_data[test_data >= 0.5] = 1.0
test_data[test_data < 0.5] = 0.0

training_batch = tf.data.Dataset.from_tensor_slices(training_data).shuffle(60000).batch(50)
test_batch = tf.data.Dataset.from_tensor_slices(test_data).shuffle(10000).batch(50)


# Create a VAE Class
class Conv_VAE(keras.Model):
    # initialization
    def __init__(self, latent_dimension):
        super(Conv_VAE, self).__init__()
        self.latent_dim = latent_dimension

        # Build encoder model
        self.encoder_model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(28, 28, 1)),
                keras.layers.Conv2D(filters=25, kernel_size=3,
                                    strides=(2, 2),
                                    activation='relu'),
                keras.layers.Conv2D(filters=50, kernel_size=3,
                                    strides=(2, 2),
                                    activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(latent_dimension + latent_dimension),
            ]
        )
        # Build decoder model
        self.decoder_model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=latent_dimension,),
                keras.layers.Dense(units=7*7*25, activation='relu'),
                keras.layers.Reshape(target_shape=(7, 7, 25)),
                keras.layers.Conv2DTranspose(filters=50, kernel_size=3,
                                             strides=(2, 2),
                                             padding='SAME',
                                             activation='relu'),
                keras.layers.Conv2DTranspose(filters=25, kernel_size=3,
                                             strides=(2, 2),
                                             padding='SAME',
                                             activation='relu'),
                keras.layers.Conv2DTranspose(filters=1, kernel_size=3,
                                             strides=(1, 1),
                                             padding='SAME'),
            ]
        )

    @tf.function
    # Sampling function for taking out samples out of encoder output
    def sampling(self, sam=None):
        if sam is None:
            sam = tf.random.normal(shape=(50, self.latent_dim))
        return self.decoder(sam, apply_sigmoid=True)

    # Encoder Function
    def encoder(self, inp):
        mean, logd = tf.split(self.encoder_model(inp),
                              num_or_size_splits=2, axis=1)
        return mean, logd

    # Reparameterization Function
    def reparameterization(self, mean, logd):
        sam = tf.random.normal(shape=mean.shape)
        return sam * tf.exp(logd * 0.5) + mean

    # Decoder function
    def decoder(self, out, apply_sigmoid=False):
        logout = self.decoder_model(out)
        if apply_sigmoid:
            probabs = tf.sigmoid(logout)
            return probabs
        return logout


def log_normal_prob_dist_func(sampling, mean_value, logd, raxis=1):
    log_2_pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(-0.5 * ((sampling - mean_value) ** 2 * tf.exp(-logd) +
                                 logd + log_2_pi), axis=raxis)

@tf.function
def loss_func(model_object, inp):
    mean_value, logd = model_object.encoder(inp)
    out = model_object.reparameterization(mean_value, logd)
    log_inp = model_object.decoder(out)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=log_inp, labels=inp)
    logp_inp_out = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logp_out = log_normal_prob_dist_func(out, 0.0, 0.0)
    logq_out_inp = log_normal_prob_dist_func(out, mean_value, logd)
    return -tf.reduce_mean(logp_inp_out + logp_out - logq_out_inp)

# Build an optimizer
optimizer = tf.keras.optimizers.Adam(0.0001)

@tf.function
def gradient_func(model_object, inp, optimizer_func):
    with tf.GradientTape() as tape:
        loss = loss_func(vae_model, inp)
    gradients = tape.gradient(loss, model_object.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_object.trainable_variables))


# Generate an image with a saved model
def generate_and_save_images(vae_model, epoch, input_data):
    preds = vae_model.sampling(input_data)
    plt.suptitle('Epoch: {}'.format(epoch))
    for i in range(preds.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(preds[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('./image/img_at_epoch{:04d}.png'.format(epoch))
    plt.pause(0.05)


# Training
epochs = 100
latent_dimension = 8
examples = 8
rand_vec = tf.random.normal(shape=[examples, latent_dimension])
vae_model = Conv_VAE(latent_dimension)

plt.ion()
fig = plt.figure(figsize=(4, 4))
generate_and_save_images(vae_model, 0, rand_vec)
for epoch in range(1, epochs+1):
    start_time = time.time()
    for x in training_batch:
        gradient_func(vae_model, x, optimizer)
    end_time = time.time()
    if epoch % 1 == 0:
        loss = keras.metrics.Mean()
        for y in test_batch:
            loss(loss_func(vae_model, y))
        elbo = -loss.result()
        # display.clear_output(wait=False)
        print('Epoch no. : {}, Test batch ELBO: {},'\
              'elapsed time for current epoch: {}'.format(epoch, elbo,
                                                          end_time - start_time))
        generate_and_save_images(vae_model, epoch, rand_vec)





