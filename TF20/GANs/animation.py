import numpy as np
import matplotlib.pyplot as plt
import imageio

fig = plt.figure()
ax = fig.add_subplot(111)

t = np.linspace(0, 2 * np.pi, 128, endpoint=False)

ims = []
for i in t:
    ax.plot(t, np.sin(t+i), color='blue', label='fwd')
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    ims.append(image)

kwargs_writer={'fps':1.0, 'quantizer:':'nq'}
imageio.mimsave('./powers.gif', ims, fps=1)
