import matplotlib.pylab as plt
import tangent as tg
from matplotlib.colors import Normalize

from dos.target_functions import *

RANGE = 4
SLOPE = 4


def f1(x, y):
    return (x / (x + y)) ** SLOPE
    # return x / ((x + y) ** SLOPE)


def f2(x, y):
    return (y / (x + y)) ** SLOPE
    # return y / ((x + y) ** SLOPE)


def h(x, y):
    return (x + y) ** SLOPE


def hdx(x, y):
    return (SLOPE * (x + y) ** (SLOPE - 1)) * (1 + x)


def hdy(x, y):
    return (SLOPE * (x + y) ** (SLOPE - 1)) * (1 + y)


def gdx(x, y):
    return SLOPE * x ** (SLOPE - 1)


def gdy(x, y):
    return SLOPE * y ** (SLOPE - 1)


def f1_dx(x, y):
    return (h(x, y) * gdx(x, y) - x ** SLOPE * hdx(x, y)) / (h(x, y) ** 2)


def f2_dy(x, y):
    return (h(x, y) * gdy(x, y) - y ** SLOPE * hdy(x, y)) / (h(x, y) ** 2)


def f1_dy(x, y):
    return (- x ** SLOPE * hdy(x, y)) / (h(x, y) ** 2)


def f2_dx(x, y):
    return (- y ** SLOPE * hdx(x, y)) / (h(x, y) ** 2)


x = y = np.linspace(0.1, RANGE, 100)
X, Y = np.meshgrid(x, y)
Z = np.add(f1(X, Y), f2(X, Y))

cmap = plt.cm.viridis
plt.figure(figsize=(8, 4))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.xlabel('$a_0$')
    plt.ylabel('$a_1$')
    cs = plt.contour(X, Y, Z, cmap=cmap)
    plt.clabel(cs, inline=1, fontsize=10)

'''
f1_dx = tg.grad(f1, verbose=1)
f2_dx = tg.grad(f2, verbose=1)
f1_dy = tg.grad(f1, wrt=[1], verbose=1)
f2_dy = tg.grad(f2, wrt=[1], verbose=1)
'''

x = y = np.linspace(.1, RANGE, 10)
X, Y = np.meshgrid(x, y)

Z = np.add(f1(X, Y), f2(X, Y))
# norm = Normalize()
# norm.autoscale(Z)

factor = .5

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.quiver(X, Y,
               f1_dx(X, Y) - (f1_dx(X, Y) - f2_dx(X, Y)) * i * factor + (f2_dy(X, Y) - f1_dy(X, Y)) * i * factor,
               f2_dy(X, Y) - (f2_dy(X, Y) - f1_dy(X, Y)) * i * factor + (f1_dx(X, Y) - f2_dx(X, Y)) * i * factor,
               Z, cmap=cmap, scale=5,
               zorder=4)

plt.show()
