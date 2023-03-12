from perlin_noise import PerlinNoise
import numpy as np
from typing import List


def constant_wind(x, y, t=0, seed=1) -> List:
    """Return a constant wind vector"""
    return [4.0, 0.0]


def linear_wind(x, y, t=0, seed=1) -> List:
    """Return a wind vector that is linear in y"""
    return [(250-y)/12.5-10, 0.0]

def perlin_noise_wind(x, y, t=0, seed=1) -> List:
    """Return a horizontal wind vector subject to perlin noise"""
    time_reduction_factor = 50
    t /= time_reduction_factor
    force = 50
    out = [0, 0]
    n_layers_of_noise = 4
    xdim, ydim = 700, 700  # dimensions of the environment (overestimated)
    for i in range(n_layers_of_noise):
        noise = PerlinNoise(octaves=int(2 ** (i + 2)), seed=seed)
        out[0] += force * noise([x / xdim, y / ydim, t]) / 2 ** (i + 1)
    return out
# to see wind map, run:

# import matplotlib.pyplot as plt
# seed = 2
# force = 50
# n_layers_of_noise = 4
# xpix, ypix = 256, 256
# pic = np.zeros((xpix, ypix))
# for i in range(n_layers_of_noise):
#     noise = PerlinNoise(octaves=int(2 ** (i + 2)), seed=seed)
#     pic += force * np.array([[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]) * 1 / 2 ** (i)
# plt.matshow(pic)
# cbar=plt.colorbar()
# cbar.set_label("Wind horizontal speed")
# plt.title("Wind map with a Perlin noise wind")
# plt.show()


class Wind:
    """Implement a spatial representation of the wind"""

    def __init__(self, wind_function=constant_wind):
        self.wind_function = wind_function
        self.seed = np.random.randint(0, 1000000000)  # setting seed for wind functions which need it

    def get_wind(self, x, y, t=0):
        return self.wind_function(x, y, t, self.seed)
