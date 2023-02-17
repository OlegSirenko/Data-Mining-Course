import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from vrosenbrock import vrosenbrock

Lx = -5  # % Левая граница для x
Rx = 5  # Правая граница для x
stepx = 0.05  # Шаг по оси x
Ly = -5  # Левая граница для y
Ry = 5  # Правая граница для y
stepy = 0.05  # Шаг по оси y

xs = np.arange(start=Lx, step=stepx, stop=Rx)
ys = np.arange(start=Ly, step=stepy, stop=Ry)

X, Y = np.meshgrid(xs, ys)
Z = vrosenbrock(X, Y)

ax = plt.axes(projection='3d')

# ax.scatter3D(X, Y, Z)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()
