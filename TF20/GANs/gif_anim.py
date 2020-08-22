import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np

fig = plt.figure()
camera = Camera(fig)
for i in range(20):
    t = plt.plot(range(i, i+5))
    plt.legend(t, [f'line{i}'])
    camera.snap()
animation = camera.animate()
animation.save('celluloid_legends.gif', writer='imagemagick')

fig2, axes = plt.subplots(2)
camera2 = Camera(fig2)
t = np.linspace(0, 2 * np.pi, 128, endpoint=False)

for i in t:
    axes[0].plot(t, np.sin(t+i), color='blue', label='fwd')
    axes[1].plot(t, np.sin(t-i), color='red', label='bwd')
    camera2.snap()

animation2 = camera2.animate()
animation2.save('celluloid_subplot.gif', writer='imagemagick')