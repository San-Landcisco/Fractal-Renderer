import numba
import numpy as np

@numba.vectorize(nopython=True)
def fast_fractal(c, depth):
    z = 0
    for step in range(depth):
        if abs(z) >= 2:
            return step

        z = z**2 + c
    return depth
