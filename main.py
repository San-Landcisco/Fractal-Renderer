import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Process, Manager, Pool
import os

from giftools import render
from frac import *


if __name__ == '__main__':
    # Domain controls for the plot
    R_range = [-2, 2]  # Real interval
    I_range = [-2, 2]  # Complex interval

    R_scale = 250  # Width number of pixels
    I_scale = abs(math.ceil(R_scale * (I_range[0]-I_range[1])/(R_range[1]-R_range[0])))

    display_trace = True
    make_gif = True
    approach = "process"

    frame_count = 1
    zoom_factor = 1

    param = 0

    DEPTH = 25 # controls number of iterations of the map
    DEPTH_SCALE = 1

    # Values for the zoom functions
    R_init = R_range[1]-R_range[0]
    I_init = I_range[1]-I_range[0]

    Del_R = R_init/2*(1-zoom_factor)
    Del_I = I_init/2*(1-zoom_factor)

    R_adjusted = [0,0]
    I_adjusted = [0,0]

    R_adjusted[0] = R_range[0]  # mutable R_range for stages of zoom
    R_adjusted[1] = R_range[1]
    I_adjusted[0] = I_range[0]  # mutable I_range for stages of zoom
    I_adjusted[1] = I_range[1]


    for frame_current in range(frame_count):
        # Shifts the window, accomplishing #zoom_factor over #frame_count frames

        DEPTH = math.ceil(DEPTH * DEPTH_SCALE)

        # _range[0] = R_range[0] + (R_init/2)*(1/frame_count)*(1-zoom_factor)
        # I_range[0] = I_range[0] + (I_init/2)*(1/frame_count)*(1-zoom_factor)

        # R_range[1] = R_range[1] - (R_init/2)*(1/frame_count)*(1-zoom_factor)
        # I_range[1] = I_range[1] - (I_init/2)*(1/frame_count)*(1-zoom_factor)

        # R_val = (R_init - math.sqrt(R_init**2-4*Del_R*R_init*frame_current/frame_count+4*Del_R**2*frame_current/frame_count))/2
        # I_val = (I_init - math.sqrt(I_init**2-4*Del_I*I_init*frame_current/frame_count+4*Del_I**2*frame_current/frame_count))/2

        R_val = (R_init - math.exp(math.log(R_init)*(frame_count-frame_current)/frame_count+math.log(R_init*zoom_factor)*frame_current/frame_count))/2
        I_val = (I_init - math.exp(math.log(I_init)*(frame_count-frame_current)/frame_count+math.log(I_init*zoom_factor)*frame_current/frame_count))/2

        R_adjusted[0] = R_range[0] + R_val
        I_adjusted[0] = I_range[0] + I_val

        R_adjusted[1] = R_range[1] - R_val
        I_adjusted[1] = I_range[1] - I_val

        if approach == "pool":
            with Pool(processes = 8) as pool:
                pool.imap_unordered(renderStrip, range(I_scale))

        if approach == "process":
            with Manager() as manager:

                data = manager.list(range(I_scale))

                cores = 8
                p = [
                    Process(target=renderStrips, args=(i, R_scale, I_scale, DEPTH, param, R_adjusted, I_adjusted, data, cores))
                    for i in range(cores)
                ]
                for p_i in p:
                    p_i.start()
                for p_i in p:
                    p_i.join()

                X = np.array(data)

        export_figure_matplotlib(X, str(frame_current), 120, 1, False)

        if display_trace:
            print('frame: ' + str(frame_current + 1))
            print(str(R_adjusted[0])+','+str(R_adjusted[1]))
            print(str(I_adjusted[0])+','+str(I_adjusted[1]))
            print("depth: " + str(DEPTH))
            print()

    if make_gif:
        directory = 'frames/'
        frames = []

        for filename in os.listdir(directory):
            frames.append((int(filename[:-4]), 'frames/' + filename))

        frames = sorted(frames)
        print(frames)

        render([y for (x, y) in frames], "test", frame_duration=1/24)
