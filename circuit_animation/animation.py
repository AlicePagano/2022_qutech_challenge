import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

xmax = 30
xmin = 3.075
ymax = 20

# Import circuit image
img = imageio.imread("circuit.png")

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(ymax, xmax)

ax = plt.axes(xlim=(0, xmax), ylim=(0, ymax))
height = 6.5
patch = plt.Rectangle((xmin,ymax-height), 1.2, height, alpha=0.5)


def init():
    patch.xy = (xmin,ymax-height)
    ax.add_patch(patch)
    return patch,

def animate(i):
    x = xmin + i%xmax
    y = (ymax-height) - (i//xmax)%ymax * (ymax/3)
    patch.xy = (x, y)
    return patch,

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=360, 
                               interval=200,
                               blit=True)

plt.imshow(img,zorder=0,  extent=[0, xmax, 0, ymax])
plt.axis('off')

anim.save('the_circuit.gif', writer='imagemagick', fps=60)
plt.show()