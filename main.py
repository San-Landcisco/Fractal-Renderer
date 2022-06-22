import matplotlib.pyplot as plt
import numpy as np
import math

from frac import *


if __name__ == '__main__':
    # Domain controls for the plot
    R_range = [-2, 2]  # Real interval
    I_range = [-2, 2]  # Complex interval

    R_scale = 512  # Width number of pixels
    I_scale = abs(math.ceil(R_scale * (I_range[0]-I_range[1])/(R_range[1]-R_range[0])))

    cam = Camera(resolution=(R_scale, I_scale), frame=(R_range, I_range))
    cam.recenter((-.10109636384562, .95628651080914), reinit=True)
    print(cam.frame)

    show_trace = True
    make_gif = False
    approach = "process"

    frame_count = 1
    zoom_factor = 1

    param = 0  # time parameter for animation

    # map thats iterated to make the fractal
    # :(
    # f = lambda x, c: (complex(abs(x.real),abs(x.imag)))**4 + c  # Burning Ship
    # f = lambda x, c: (x.conjugate())**2+c  # Tricorn / Mandelbar

    DEPTH = 100  # controls number of iterations of the map
    DEPTH_SCALE = 1

    zoomer = Animation(cam, frame_count=480, zoom_factor=0.001, depth=128, depth_scale=1.001)
    zoomer.animate()

    #cam.capture_frame()
