"""
Unconditional GAN
Tensorflow 2.0
The animation shows how generator learns the input data distribution through adversarial training
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from celluloid import Camera


# This is the real data distribution which generator tries to learn
def sample_dataset():
    dataset_shape = (2000, 1)
    return tf.random.normal(mean=10., shape=dataset_shape,
                            stddev=0.1, dtype=tf.float32)

# counts, bin, ignored = plt.hist(sample_dataset().numpy(), 100)
# axes = plt.gca()
# axes.set_xlim([-1, 11])
# axes.set_ylim([0, 60])
#plt.show()


# Define the generator
# task is to map a noise signal into a data that resembles input data distribution
def generator(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
    net = tf.keras.layers.Dense(units=1, name="G")(net)
    G = tf.keras.Model(inputs=inputs, outputs=net)
    return G


# Define the discriminator
# differentiate between true data and fake data (generated by Generator)
def discriminator(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=1, name="D")(net)
    D = tf.keras.Model(inputs=inputs, outputs=net)
    return D


input_shape = (1, )
D = discriminator(input_shape)

# Arbitrary set the shape of noise prior
latent_space_shape = (100, )
G = generator(latent_space_shape)

# discriminator loss
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# max_D E_x [log(D(x))] + E_z [log (1-D(G(z)))]
def d_loss(d_real, d_fake):
    return bce(tf.ones_like(d_real), d_real) + \
           bce(tf.zeros_like(d_fake), d_fake)


# max_G E_z [log(D(G(z)))]
def g_loss(generated_output):
    return bce(tf.ones_like(generated_output),
               generated_output)


# Adversarial Training
def train():
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(1e-5)

    @tf.function
    def train_step():
        with tf.GradientTape(persistent=True) as tape:
            real_data = sample_dataset()
            noise_vector = tf.random.normal(
                mean=0, stddev=1,
                shape=(real_data.shape[0], latent_space_shape[0])
            )
            # sample from the generator
            fake_data = G(noise_vector)
            # compute D loss
            d_fake_data = D(fake_data)
            d_real_data = D(real_data)
            d_loss_value = d_loss(d_real_data, d_fake_data)
            # compute G loss
            g_loss_value = g_loss(d_fake_data)
        # compute gradients
        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        # deleting tape, since we defined it as persistent (we used it twice)
        del tape
        optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
        optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))
        return real_data, fake_data, g_loss_value, d_loss_value


    # visualize distribution
    #plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    camera = Camera(fig)
    for step in range(40000):
        real_data, fake_data, g_loss_value, d_loss_value = train_step()
        if step % 200 == 0:
            print("G loss: ", g_loss_value.numpy(),
                  "D loss: ", d_loss_value.numpy(), "step :", step)
            #plt.cla()
            # sample 5000 values from the Generator and draw the histogram
            ax.hist(fake_data.numpy(), 100, color='blue', label='fake')
            ax.hist(real_data.numpy(), 100, color='red', label='real')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            textstr = f"step={step}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes = plt.gca()
            axes.set_xlim([-1, 11])
            axes.set_ylim([0, 60])
            plt.legend(['fake', 'real'], loc='center left')
            camera.snap()
            #plt.cla() # clear current axis
            #plt.savefig('final_hist.png')
            #plt.pause(0.5)
    animation = camera.animate()
    animation.save('celluloid_minimal.gif', writer='imagemagick')
##############
train()




