import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return x**2 + y**2 / 4


x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
xmesh, ymesh = np.meshgrid(x, y)
z = f(xmesh, ymesh)

plt.contour(x, y, z, colors="k", levels=[1, 2, 3, 4, 5])
plt.show()
